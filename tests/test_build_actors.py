"""Testes do montador da base — sem tocar na internet.

A maior parte usa dublês para exercitar a mecânica: escolha das URLs,
retentativa, merge com a base existente e relatório final. No fim há um teste
de ponta a ponta contra um servidor local que imita a API da Wikipédia, que
percorre o caminho real (montar URL, ler JSON, baixar imagem, achar o rosto,
gravar a base) e por isso carrega o modelo.
"""

import urllib.error

import numpy as np
import pytest

from app import config
from scripts import build_actors as build


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTORS_NPZ", tmp_path / "actors.npz")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "THUMB_DIR", tmp_path / "thumbs")
    return tmp_path


def unit(seed: int) -> np.ndarray:
    vec = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return vec / np.linalg.norm(vec)


def write_base(entries):
    build.save([{"name": n, "thumb": t, "vector": v} for n, t, v in entries])


# --- slug e nomes ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Tom Hanks", "tom-hanks"),
        ("Fernanda Torres", "fernanda-torres"),
        ("Taís Araújo", "tais-araujo"),
        ("Robert Downey Jr.", "robert-downey-jr"),
        ("Lupita Nyong'o", "lupita-nyongo"),
        ("Timothée Chalamet", "timothee-chalamet"),
    ],
)
def test_slugify_makes_safe_filenames(name, expected):
    assert build.slugify(name) == expected


# --- escolha das URLs -----------------------------------------------------


def test_manual_urls_win_over_wikipedia(monkeypatch):
    monkeypatch.setitem(build.PHOTO_URLS, "Fulano", ["https://exemplo.com/a.jpg"])
    monkeypatch.setattr(build, "_main_photo", lambda a: pytest.fail("não devia consultar a API"))
    assert build.photo_urls("Fulano", 4) == ["https://exemplo.com/a.jpg"]


def test_manual_urls_respect_the_limit(monkeypatch):
    monkeypatch.setitem(build.PHOTO_URLS, "Fulano", [f"https://e.com/{i}.jpg" for i in range(9)])
    assert len(build.photo_urls("Fulano", 3)) == 3


def test_main_photo_comes_first(monkeypatch):
    monkeypatch.setattr(build, "_main_photo", lambda a: ["https://e.com/retrato.jpg"])
    monkeypatch.setattr(build, "_gallery", lambda a: ["https://e.com/outra.jpg"])
    assert build.photo_urls("Fulano", 4)[0] == "https://e.com/retrato.jpg"


def test_a_broken_source_does_not_stop_the_others(monkeypatch):
    def falha(_):
        raise urllib.error.URLError("sem rede")

    monkeypatch.setattr(build, "_main_photo", falha)
    monkeypatch.setattr(build, "_gallery", lambda a: ["https://e.com/b.jpg"])
    assert build.photo_urls("Fulano", 4) == ["https://e.com/b.jpg"]


def test_duplicate_urls_are_dropped(monkeypatch):
    monkeypatch.setattr(build, "_main_photo", lambda a: ["https://e.com/x.jpg"])
    monkeypatch.setattr(build, "_gallery", lambda a: ["https://e.com/x.jpg", "https://e.com/y.jpg"])
    assert build.photo_urls("Fulano", 4) == ["https://e.com/x.jpg", "https://e.com/y.jpg"]


# --- retentativa ----------------------------------------------------------


def test_get_retries_on_a_temporary_failure(monkeypatch):
    calls = []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("caiu")
        raise SystemExit("chegou na terceira")  # marcador: houve retentativa

    monkeypatch.setattr(build.time, "sleep", lambda s: None)
    monkeypatch.setattr(build.urllib.request, "urlopen", flaky)
    with pytest.raises(SystemExit):
        build._get("https://e.com/a.jpg")
    assert len(calls) == 3


def test_get_does_not_retry_a_404(monkeypatch):
    calls = []

    def not_found(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(build.time, "sleep", lambda s: None)
    monkeypatch.setattr(build.urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        build._get("https://e.com/a.jpg")
    assert len(calls) == 1  # insistir num 404 só faria o build demorar mais


# --- gravação e merge -----------------------------------------------------


def test_save_and_load_roundtrip(base):
    vector = unit(1)
    write_base([("Tom Hanks", "tom-hanks.jpg", vector)])

    loaded = build.load_existing()
    assert set(loaded) == {"Tom Hanks"}
    assert loaded["Tom Hanks"]["thumb"] == "tom-hanks.jpg"
    assert loaded["Tom Hanks"]["vector"] == pytest.approx(vector)


def test_load_existing_is_empty_without_a_database(base):
    assert build.load_existing() == {}


def test_merge_keeps_who_was_already_there(base):
    write_base([("Tom Hanks", "a.jpg", unit(1)), ("Alice Braga", "b.jpg", unit(2))])

    entries = build.load_existing()
    entries["Wagner Moura"] = {"name": "Wagner Moura", "thumb": "c.jpg", "vector": unit(3)}
    build.save(list(entries.values()))

    assert set(build.load_existing()) == {"Tom Hanks", "Alice Braga", "Wagner Moura"}


def test_merge_updates_a_repeated_actor_without_duplicating(base):
    write_base([("Tom Hanks", "velha.jpg", unit(1))])

    entries = build.load_existing()
    novo = unit(9)
    entries["Tom Hanks"] = {"name": "Tom Hanks", "thumb": "nova.jpg", "vector": novo}
    build.save(list(entries.values()))

    loaded = build.load_existing()
    assert len(loaded) == 1
    assert loaded["Tom Hanks"]["thumb"] == "nova.jpg"
    assert loaded["Tom Hanks"]["vector"] == pytest.approx(novo)


def test_saved_database_is_readable_by_the_server(base):
    from app import matching

    write_base([("Tom Hanks", "a.jpg", unit(1)), ("Alice Braga", "b.jpg", unit(2))])
    matching._db = None
    db = matching.load()
    matching._db = None

    assert len(db) == 2
    assert len(db.rank(unit(1), top_n=2)) == 2


# --- relatório ------------------------------------------------------------


def test_report_lists_who_was_left_out(base, caplog):
    with caplog.at_level("INFO", logger="build_actors"):
        build.report(total=10, before=set(), failed=["Fulano de Tal"])
    text = caplog.text
    assert "Fulano de Tal" in text
    assert "photo_urls.py" in text  # aponta o caminho da correção


def test_report_is_quiet_when_everything_worked(base, caplog):
    with caplog.at_level("INFO", logger="build_actors"):
        build.report(total=10, before=set(), failed=[])
    assert "Todos os atores da lista entraram." in caplog.text
    assert "Ficaram de fora" not in caplog.text


# --- caminho completo contra uma Wikipédia de mentira ---------------------


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
                        "pages": [{"original": {"source": f"http://127.0.0.1:{porta}/foto.jpg"}}]
                    }
                }
            else:
                corpo = {
                    "query": {
                        "pages": [
                            {
                                "title": "File:Retrato.jpg",
                                "imageinfo": [{"url": f"http://127.0.0.1:{porta}/foto.jpg"}],
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


def test_end_to_end_build_against_a_fake_wikipedia(base, fake_wikipedia, monkeypatch):
    pytest.importorskip("torch")
    from app import face

    try:
        face.warmup()
    except Exception as exc:
        pytest.skip(f"modelo indisponível: {exc}")

    monkeypatch.setattr(build, "API", fake_wikipedia)
    monkeypatch.setattr(build, "ACTORS", ["Ator Um", "Ator Dois", "Ator Desconhecido"])
    monkeypatch.setattr("sys.argv", ["build_actors", "--per-actor", "1"])

    assert build.main() == 0

    entries = build.load_existing()
    assert set(entries) == {"Ator Um", "Ator Dois"}  # o sem foto ficou de fora
    for entry in entries.values():
        assert entry["vector"].shape == (512,)
        assert np.linalg.norm(entry["vector"]) == pytest.approx(1.0, abs=1e-4)
        assert (config.THUMB_DIR / entry["thumb"]).exists()


def test_merge_adds_to_an_existing_database(base, fake_wikipedia, monkeypatch):
    pytest.importorskip("torch")
    from app import face

    try:
        face.warmup()
    except Exception as exc:
        pytest.skip(f"modelo indisponível: {exc}")

    write_base([("Ator Antigo", "antigo.jpg", unit(7))])

    monkeypatch.setattr(build, "API", fake_wikipedia)
    monkeypatch.setattr(build, "ACTORS", ["Ator Novo"])
    monkeypatch.setattr("sys.argv", ["build_actors", "--per-actor", "1", "--merge"])

    assert build.main() == 0
    assert set(build.load_existing()) == {"Ator Antigo", "Ator Novo"}


def test_build_fails_loudly_when_nothing_could_be_fetched(base, fake_wikipedia, monkeypatch):
    monkeypatch.setattr(build, "API", fake_wikipedia)
    monkeypatch.setattr(build, "ACTORS", ["Ator Desconhecido"])
    monkeypatch.setattr("sys.argv", ["build_actors"])
    monkeypatch.setattr(build.face, "warmup", lambda: None)

    assert build.main() == 1
    assert not config.ACTORS_NPZ.exists()  # base antiga não é destruída
