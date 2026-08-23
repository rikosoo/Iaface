import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app import config

# Foto de rosto usada nos testes de ponta a ponta. Vem do repositório do
# facenet-pytorch para não guardar imagem de pessoa real neste projeto; fica
# em cache depois do primeiro download.
PHOTO_URL = (
    "https://raw.githubusercontent.com/timesler/facenet-pytorch/master"
    "/data/test_images/angelina_jolie/1.jpg"
)
PHOTO_CACHE = Path(__file__).parent / "fixtures" / "face.jpg"


@pytest.fixture(scope="session")
def photo() -> bytes:
    if PHOTO_CACHE.exists():
        return PHOTO_CACHE.read_bytes()

    PHOTO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(PHOTO_URL, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"sem foto de teste (offline): {exc}")

    PHOTO_CACHE.write_bytes(data)
    return data


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTORS_NPZ", tmp_path / "actors.npz")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "THUMB_DIR", tmp_path / "thumbs")
    return tmp_path


@pytest.fixture
def fake_wikipedia(photo):
    """Sobe um servidor local que responde como a API da Wikipédia.

    É o único jeito de exercitar de verdade o caminho completo — montar a
    URL, ler o JSON, baixar a imagem, achar o rosto e gravar a base — sem
    depender da internet no momento do teste.
    """
    import json as json_mod
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # sem poluir a saída do pytest

        def _send(self, body: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/foto.jpg":
                return self._send(photo, "image/jpeg")

            query = parse_qs(url.query)
            titulo = query.get("titles", [""])[0]
            porta = self.server.server_address[1]

            if "Desconhecido" in titulo:  # ator sem foto no acervo
                return self._send(json_mod.dumps({"query": {"pages": []}}).encode(), "text/json")

            if query.get("prop") == ["pageimages"]:
                corpo = {
                    "query": {
                        "pages": [{"thumbnail": {"source": f"http://127.0.0.1:{porta}/foto.jpg"}}]
                    }
                }
            else:
                corpo = {
                    "query": {
                        "pages": [
                            {
                                "title": "File:Retrato.jpg",
                                "imageinfo": [{"thumburl": f"http://127.0.0.1:{porta}/foto.jpg"}],
                            },
                            # Ruído que o filtro tem que descartar:
                            {"title": "File:Commons-logo.svg", "imageinfo": []},
                        ]
                    }
                }
            self._send(json_mod.dumps(corpo).encode(), "text/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/w/api.php"
    server.shutdown()
