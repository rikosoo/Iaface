"""Servidor local do Iaface.

    uvicorn app.server:app --port 8000

Abra http://localhost:8000 — o navegador só libera a câmera em localhost ou
HTTPS.
"""

from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app import config, face, limits, matching, style

log = logging.getLogger("iaface")


class MatchOut(BaseModel):
    name: str
    thumb: str | None
    similarity: float
    percent: float
    label: str


class QualityOut(BaseModel):
    face_pixels: int
    sharpness: float
    brightness: float
    detection_prob: float
    warnings: list[str]


class ColorOut(BaseModel):
    name: str
    hex: str


class StyleOut(BaseModel):
    skin_hex: str
    hair_hex: str | None
    eyes_hex: str | None
    undertone: str
    depth: str
    contrast: str
    season: str
    idea: str
    metals: str
    contrast_tip: str
    palette: list[ColorOut]
    avoid: list[ColorOut]
    pieces: list[str]
    confidence: str
    notes: list[str]


class MatchResponse(BaseModel):
    matches: list[MatchOut]
    verdict: str
    quality: QualityOut
    face_crop: str  # data URI do recorte que foi analisado
    style: StyleOut | None


class StatusResponse(BaseModel):
    ready: bool
    actors: int
    # O front usa isto para exigir o consentimento antes de abrir a câmera.
    public: bool
    busy: bool


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Carregar os pesos aqui evita que o primeiro usuário pague a espera.
    face.warmup()
    try:
        log.info("base carregada com %d atores", len(matching.load()))
    except matching.ActorDatabaseMissing:
        log.warning("base de atores ausente — rode: python -m scripts.build_actors")
    yield


app = FastAPI(title="Iaface", docs_url=None, redoc_url=None, lifespan=lifespan)

fila = limits.Semaforo(config.MAX_CONCURRENCY, config.QUEUE_TIMEOUT)
limite_ip = limits.LimitePorIP(config.RATE_LIMIT, config.RATE_WINDOW)

if config.CORS_ORIGINS:
    # Só entra quando a API é chamada de outro domínio. Embutido em iframe a
    # origem é a mesma, e aí a lista fica vazia — que é o padrão mais fechado.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "X-Iaface-Consent"],
        max_age=3600,
    )


def _actor_db() -> matching.ActorDatabase:
    try:
        return matching.load()
    except matching.ActorDatabaseMissing:
        raise HTTPException(
            status_code=503,
            detail="Base de atores ausente. Rode: python -m scripts.build_actors",
        )


def _data_uri(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/privacidade")
def privacy() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "privacidade.html")


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    ocupado = fila.livres == 0
    try:
        atores = len(matching.load())
    except matching.ActorDatabaseMissing:
        return StatusResponse(ready=False, actors=0, public=config.PUBLIC, busy=ocupado)
    return StatusResponse(ready=True, actors=atores, public=config.PUBLIC, busy=ocupado)


@app.post("/api/match", response_model=MatchResponse)
async def match(request: Request, response: Response) -> MatchResponse:
    """Analisa a foto e devolve o resultado — sem gravar nada em disco.

    A imagem chega como corpo cru da requisição, e não como multipart de
    formulário, de propósito: o parser multipart do Starlette derrama uploads
    acima de 1 MB num arquivo temporário no disco. Foto de rosto é dado
    biométrico (GDPR art. 9), então ela não pode encostar no disco em momento
    nenhum. Lendo o corpo direto, os bytes existem só na memória do processo.
    """
    # Sem no-store, o resultado (que inclui o recorte do rosto) ficaria no
    # cache do navegador depois que a pessoa sair da página.
    response.headers["Cache-Control"] = "no-store"

    tamanho = request.headers.get("content-length")
    if tamanho and int(tamanho) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagem grande demais (máx. 12 MB).")

    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Imagem vazia.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagem grande demais (máx. 12 MB).")

    db = _actor_db()

    if config.PUBLIC:
        # Consentimento explícito é a base legal para tratar dado do Art. 9.
        # O front só manda este cabeçalho depois que a pessoa marca a caixa.
        if request.headers.get("x-iaface-consent") != "granted":
            raise HTTPException(
                status_code=403,
                detail="É preciso autorizar a análise da foto antes de continuar.",
            )
        limite_ip.registrar(limits.ip_do_cliente(request))

    # A inferência é CPU pura e síncrona. Rodá-la direto aqui travaria o laço
    # de eventos e, com ele, todas as outras requisições — inclusive as
    # estáticas. Medido: /api/status ia de 2ms para 776ms durante uma análise.
    async with fila:
        return await run_in_threadpool(_analisar, raw, db)


def _analisar(raw: bytes, db: matching.ActorDatabase) -> MatchResponse:
    """Todo o trabalho pesado, rodando fora do laço de eventos."""
    try:
        image = face.load_image(raw)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Não consegui ler essa imagem.")

    try:
        detected = face.detect(image)
    except face.NoFaceFound:
        raise HTTPException(
            status_code=422,
            detail="Não achei um rosto na foto. Aproxime o rosto, melhore a luz e tente de novo.",
        )

    vector = face.embed(detected)
    matches = db.rank(vector)
    profile = style.analyze(detected.image, detected.box, detected.landmarks)

    return MatchResponse(
        matches=[MatchOut(**vars(m)) for m in matches],
        verdict=matching.verdict(matches),
        quality=QualityOut(**vars(detected.quality)),
        face_crop=_data_uri(detected.crop),
        style=StyleOut(**asdict(profile)) if profile else None,
    )


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
