"""Servidor local do Iaface.

    uvicorn app.server:app --reload --port 8000

Abra http://localhost:8000 — a câmera só é liberada pelo navegador em
localhost ou HTTPS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import face

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
ACTORS_NPZ = ROOT / "data" / "actors.npz"

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
TOP_N = 3

app = FastAPI(title="Iaface", docs_url=None, redoc_url=None)

_db: dict | None = None


def actor_db() -> dict:
    """Carrega a base de atores (uma vez) gerada por scripts/build_actors.py."""
    global _db
    if _db is None:
        if not ACTORS_NPZ.exists():
            raise HTTPException(
                status_code=503,
                detail="Base de atores ausente. Rode: python -m scripts.build_actors",
            )
        data = np.load(ACTORS_NPZ, allow_pickle=False)
        _db = {
            "names": [str(n) for n in data["names"]],
            "thumbs": [str(t) for t in data["thumbs"]],
            "vectors": data["vectors"].astype(np.float32),
        }
    return _db


@app.on_event("startup")
def _startup() -> None:
    face.warmup()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    try:
        db = actor_db()
    except HTTPException:
        return {"ready": False, "actors": 0}
    return {"ready": True, "actors": len(db["names"])}


@app.post("/api/match")
async def match(photo: UploadFile = File(...)) -> dict:
    raw = await photo.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Imagem vazia.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagem grande demais (máx. 12 MB).")

    db = actor_db()

    try:
        vector = face.embed_bytes(raw)
    except face.NoFaceFound:
        raise HTTPException(
            status_code=422,
            detail="Não achei um rosto na foto. Aproxime o rosto e melhore a luz.",
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Não consegui ler essa imagem.")

    # Vetores já são normalizados, então o produto escalar é o cosseno.
    scores = db["vectors"] @ vector
    order = np.argsort(-scores)[:TOP_N]

    return {
        "matches": [
            {
                "name": db["names"][i],
                "thumb": f"/static/actors/{db['thumbs'][i]}" if db["thumbs"][i] else None,
                "similarity": round(float(scores[i]), 4),
                # O cosseno útil vive entre ~0.2 e ~0.9; esticar essa faixa dá
                # uma porcentagem que corresponde melhor ao que se vê na tela.
                "percent": round(float(np.clip((scores[i] - 0.15) / 0.65, 0, 1)) * 100, 1),
            }
            for i in order
        ]
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
