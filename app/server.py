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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app import config, face, matching, style

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


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    try:
        return StatusResponse(ready=True, actors=len(matching.load()))
    except matching.ActorDatabaseMissing:
        return StatusResponse(ready=False, actors=0)


@app.post("/api/match", response_model=MatchResponse)
async def match(photo: UploadFile = File(...)) -> MatchResponse:
    raw = await photo.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Imagem vazia.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagem grande demais (máx. 12 MB).")

    db = _actor_db()

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
