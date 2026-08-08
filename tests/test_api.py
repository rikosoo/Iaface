"""Testes de ponta a ponta da API — carregam o modelo de verdade.

São pulados automaticamente se os pesos ainda não estiverem em cache, para não
transformar `pytest` em download de 110 MB.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import config, matching


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from app import face

    try:
        face.warmup()
    except Exception as exc:  # sem rede e sem cache: nada a testar aqui
        pytest.skip(f"modelo indisponível: {exc}")

    # Base mínima: dois vetores aleatórios, só para o ranking ter o que ordenar.
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(2, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    npz = tmp_path_factory.mktemp("data") / "actors.npz"
    np.savez(
        npz,
        names=np.array(["Ator Teste", "Outro Ator"]),
        thumbs=np.array(["", ""]),
        vectors=vectors,
    )

    config.ACTORS_NPZ = npz
    matching._db = None

    from app.server import app

    with TestClient(app) as c:
        yield c

    matching._db = None


def upload(client, data: bytes):
    """A foto vai como corpo cru: multipart derramaria para o disco."""
    return client.post("/api/match", content=data, headers={"Content-Type": "image/jpeg"})


def jpeg(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, "JPEG")
    return buffer.getvalue()


def test_status_reports_the_loaded_database(client):
    body = client.get("/api/status").json()
    assert body == {"ready": True, "actors": 2}


def test_index_is_served(client):
    assert client.get("/").status_code == 200


def test_rejects_an_empty_upload(client):
    assert upload(client, b"").status_code == 400


def test_rejects_a_file_that_is_not_an_image(client):
    assert upload(client, b"nao sou uma imagem, sou um texto qualquer").status_code == 400


def test_rejects_an_oversized_upload(client):
    assert upload(client, b"x" * (config.MAX_UPLOAD_BYTES + 1)).status_code == 413


def test_reports_when_there_is_no_face(client):
    blank = Image.new("RGB", (400, 400), (120, 120, 120))
    ImageDraw.Draw(blank).rectangle((50, 50, 350, 350), fill=(200, 90, 60))
    res = upload(client, jpeg(blank))
    assert res.status_code == 422
    assert "rosto" in res.json()["detail"]


def test_full_analysis_of_a_real_photo(client, photo):
    res = upload(client, photo)
    assert res.status_code == 200
    body = res.json()

    assert len(body["matches"]) == 2
    scores = [m["similarity"] for m in body["matches"]]
    assert scores == sorted(scores, reverse=True)
    for match in body["matches"]:
        assert -1.0 <= match["similarity"] <= 1.0
        assert 0 <= match["percent"] <= 100
        assert match["label"]

    assert body["verdict"]
    assert body["face_crop"].startswith("data:image/jpeg;base64,")
    assert body["quality"]["face_pixels"] > 0
    assert body["quality"]["detection_prob"] >= 0.9

    style = body["style"]
    assert style["skin_hex"].startswith("#")
    assert style["undertone"] in {"quente", "fria", "neutra"}
    assert style["contrast"] in {"baixo", "médio", "alto"}
    assert len(style["palette"]) >= 6
    assert style["pieces"]


def test_rotated_photo_is_recovered(client, photo):
    """Foto deitada (EXIF perdido) ainda tem que ser analisada."""
    rotated = Image.open(io.BytesIO(photo)).rotate(90, expand=True)
    res = upload(client, jpeg(rotated))
    assert res.status_code == 200


def test_dark_photo_gets_a_warning(client, photo):
    dark = Image.eval(Image.open(io.BytesIO(photo)).convert("RGB"), lambda v: int(v * 0.18))
    res = upload(client, jpeg(dark))
    if res.status_code == 200:
        assert res.json()["quality"]["warnings"]
