"""Testes que sustentam a promessa da página de privacidade.

A alegação "nada é gravado em disco" só vale alguma coisa se for verificável.
Aqui a análise roda de verdade e o teste observa o sistema de arquivos e as
chamadas de abertura de arquivo temporário enquanto isso acontece.
"""

import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, matching


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("torch")
    from app import face

    try:
        face.warmup()
    except Exception as exc:
        pytest.skip(f"modelo indisponível: {exc}")

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(2, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    npz = tmp_path_factory.mktemp("data") / "actors.npz"
    np.savez(npz, names=np.array(["A", "B"]), thumbs=np.array(["", ""]), vectors=vectors)

    config.ACTORS_NPZ = npz
    matching._db = None

    from app.server import app

    with TestClient(app) as c:
        yield c

    matching._db = None


def analisar(client, photo):
    return client.post(
        "/api/match", content=photo, headers={"Content-Type": "application/octet-stream"}
    )


def arquivos_em(pastas) -> set:
    encontrados = set()
    for pasta in pastas:
        if pasta.exists():
            encontrados |= {p for p in pasta.rglob("*") if p.is_file()}
    return encontrados


def test_the_analysis_writes_nothing_to_disk(client, photo, tmp_path):
    """A promessa central: analisar uma foto não deixa arquivo nenhum."""
    from pathlib import Path

    vigiadas = [Path(tempfile.gettempdir()), config.ROOT, Path.home() / ".cache"]

    antes = arquivos_em(vigiadas)
    assert analisar(client, photo).status_code == 200
    depois = arquivos_em(vigiadas)

    novos = depois - antes
    assert not novos, f"a análise criou arquivos: {sorted(str(p) for p in novos)[:5]}"


def test_no_temporary_file_is_opened_during_the_analysis(client, photo, monkeypatch):
    """Arquivo temporário que nasce e morre não apareceria no teste acima.

    O parser multipart do Starlette faz exatamente isso com upload acima de
    1 MB — é o motivo de o endpoint ler o corpo cru da requisição.
    """
    aberturas = []

    original = tempfile.SpooledTemporaryFile
    monkeypatch.setattr(
        tempfile,
        "SpooledTemporaryFile",
        lambda *a, **kw: (aberturas.append(1), original(*a, **kw))[1],
    )
    original_named = tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        lambda *a, **kw: (aberturas.append(1), original_named(*a, **kw))[1],
    )

    assert analisar(client, photo).status_code == 200
    assert not aberturas, "algo abriu um arquivo temporário durante a análise"


def test_a_large_photo_also_stays_in_memory(client, photo, tmp_path):
    """Acima de 1 MB é justamente onde o caminho antigo derramava para o disco."""
    from pathlib import Path

    import io

    from PIL import Image

    grande = io.BytesIO()
    Image.open(io.BytesIO(photo)).convert("RGB").resize((2400, 2400)).save(
        grande, "JPEG", quality=98
    )
    dados = grande.getvalue()
    assert len(dados) > 1024 * 1024  # senão o teste não prova nada

    vigiadas = [Path(tempfile.gettempdir()), config.ROOT]
    antes = arquivos_em(vigiadas)
    resposta = analisar(client, dados)
    assert resposta.status_code in (200, 422)  # o redimensionamento pode perder o rosto
    assert not arquivos_em(vigiadas) - antes


def test_the_response_forbids_browser_caching(client, photo):
    """O resultado carrega o recorte do rosto; não pode ficar no cache."""
    resposta = analisar(client, photo)
    assert resposta.headers["cache-control"] == "no-store"


def test_an_oversized_body_is_refused_by_the_header_before_being_read(client):
    resposta = client.post(
        "/api/match",
        content=b"x" * 32,
        headers={"Content-Length": str(config.MAX_UPLOAD_BYTES + 1)},
    )
    assert resposta.status_code == 413


def test_the_response_never_echoes_the_original_photo(client, photo):
    """A resposta traz o recorte analisado, não a imagem que foi enviada."""
    corpo = analisar(client, photo).json()
    import base64

    recorte = base64.b64decode(corpo["face_crop"].split(",", 1)[1])
    assert recorte != photo
    assert len(recorte) < len(photo)


def test_the_privacy_page_is_served(client):
    resposta = client.get("/privacidade")
    assert resposta.status_code == 200
    texto = resposta.text
    assert "Artigo 9" in texto
    assert "memória" in texto


def test_the_upload_screen_states_the_privacy_promise(client):
    """A promessa tem que estar na tela de captura, não só escondida numa página."""
    home = client.get("/").text
    assert "biométrico" in home or "biometrico" in home
    assert "/privacidade" in home
