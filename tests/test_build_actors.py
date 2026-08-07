"""Testes do montador da base — sem tocar na internet.

A maior parte usa dublês para exercitar a mecânica: escolha das URLs,
retentativa, merge com a base existente e relatório final. No fim há um teste
de ponta a ponta contra um servidor local que imita a API da Wikipédia, que
percorre o caminho real (montar URL, ler JSON, baixar imagem, achar o rosto,
gravar a base) e por isso carrega o modelo.
"""

import urllib.error
from pathlib import Path

import numpy as np
import pytest

from app import config
from scripts import build_actors as build


@pytest.fixture(autouse=True)
def sem_espera(monkeypatch):
    """Desliga o limitador de ritmo: teste não precisa ser educado com ninguém."""
    monkeypatch.setattr(build, "MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr(build.time, "sleep", lambda s: None)


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

    monkeypatch.setattr(build.urllib.request, "urlopen", flaky)
    with pytest.raises(SystemExit):
        build._get("https://e.com/a.jpg")
    assert len(calls) == 3


def test_get_does_not_retry_a_404(monkeypatch):
    calls = []

    def not_found(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

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


# --- política de robô da Wikimedia ----------------------------------------


def test_user_agent_identifies_the_project_with_a_contact():
    """Sem contato no User-Agent, a Wikimedia responde 429 por política."""
    assert "Iaface" in build.UA
    assert "http" in build.UA  # a URL do projeto é o contato


def test_a_429_is_retried_honoring_retry_after(monkeypatch):
    esperas = []
    monkeypatch.setattr(build.time, "sleep", esperas.append)

    tentativas = []

    def limitado(req, timeout):
        tentativas.append(1)
        if len(tentativas) < 2:
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {"Retry-After": "7"}, None)
        raise SystemExit("passou na segunda")

    monkeypatch.setattr(build.urllib.request, "urlopen", limitado)
    with pytest.raises(SystemExit):
        build._get("https://e.com/a.jpg")

    assert len(tentativas) == 2
    assert 7 in esperas  # esperou o que o servidor pediu, não o nosso palpite


def test_retry_after_falls_back_when_the_header_is_absent(monkeypatch):
    esperas = []
    monkeypatch.setattr(build.time, "sleep", esperas.append)

    def limitado(req, timeout):
        raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(build.urllib.request, "urlopen", limitado)
    with pytest.raises(urllib.error.HTTPError):
        build._get("https://e.com/a.jpg")

    # Espera crescente: um 429 pede recuo de verdade, não meio segundo.
    assert esperas == sorted(esperas) and esperas[0] >= 5


def test_retry_after_ignores_a_date_header():
    exc = urllib.error.HTTPError("u", 429, "", {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None)
    assert build._retry_after(exc) is None


def test_retry_after_is_capped():
    exc = urllib.error.HTTPError("u", 429, "", {"Retry-After": "99999"}, None)
    assert build._retry_after(exc) == 120.0


def test_rate_limiter_spaces_requests_apart(monkeypatch):
    dormidas = []
    relogio = iter([0.0, 0.0, 0.05, 0.05])  # segunda chamada logo após a primeira

    monkeypatch.setattr(build, "MIN_REQUEST_INTERVAL", 1.0)
    monkeypatch.setattr(build, "_last_request", 0.0)
    monkeypatch.setattr(build.time, "sleep", dormidas.append)
    monkeypatch.setattr(build.time, "monotonic", lambda: next(relogio))

    build._wait_turn()
    build._wait_turn()

    assert dormidas and dormidas[-1] == pytest.approx(0.95, abs=0.01)


def test_gallery_takes_the_thumbnail_and_skips_the_original(monkeypatch):
    resposta = {
        "query": {
            "pages": [
                {
                    "title": "File:Retrato.jpg",
                    "imageinfo": [{"thumburl": "https://e.com/800px-x.jpg", "url": "https://e.com/x.jpg"}],
                },
                # Sem miniatura disponível: não vale buscar o original.
                {"title": "File:Outra.jpg", "imageinfo": [{"url": "https://e.com/original.jpg"}]},
            ]
        }
    }
    monkeypatch.setattr(build, "_api", lambda params: resposta)
    assert build._gallery("Fulano") == ["https://e.com/800px-x.jpg"]


def test_main_photo_asks_for_a_thumbnail(monkeypatch):
    pedidos = {}

    def espiao(params):
        pedidos.update(params)
        return {"query": {"pages": [{"thumbnail": {"source": "https://e.com/800px-x.jpg"}}]}}

    monkeypatch.setattr(build, "_api", espiao)
    assert build._main_photo("Fulano") == ["https://e.com/800px-x.jpg"]
    assert pedidos["piprop"] == "thumbnail"
    assert pedidos["pithumbsize"] == str(build.THUMB_WIDTH)


def test_url_with_query_string_is_still_recognized_as_an_image(monkeypatch):
    # A API devolve as URLs com ?utm_source=... grudado no fim.
    com_query = "https://e.com/800px-x.jpg?utm_source=en.wikipedia.org"
    monkeypatch.setattr(build, "_api", lambda p: {"query": {"pages": [{"thumbnail": {"source": com_query}}]}})
    assert build._main_photo("Fulano") == [com_query]


# --- pasta de fotos do usuário --------------------------------------------


def foto_vazia(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"nao importa o conteudo aqui")


def test_local_actors_reads_names_from_subfolders(tmp_path):
    foto_vazia(tmp_path / "Tom Hanks" / "1.jpg")
    foto_vazia(tmp_path / "Fernanda Torres" / "premiere.png")
    assert build.local_actors(tmp_path) == ["Fernanda Torres", "Tom Hanks"]


def test_local_actors_reads_names_from_loose_files(tmp_path):
    foto_vazia(tmp_path / "Alice Braga.jpg")
    foto_vazia(tmp_path / "Wagner Moura.jpeg")
    assert build.local_actors(tmp_path) == ["Alice Braga", "Wagner Moura"]


def test_local_actors_mixes_both_layouts(tmp_path):
    foto_vazia(tmp_path / "Tom Hanks" / "1.jpg")
    foto_vazia(tmp_path / "Alice Braga.jpg")
    assert build.local_actors(tmp_path) == ["Alice Braga", "Tom Hanks"]


def test_local_actors_ignores_junk(tmp_path):
    foto_vazia(tmp_path / "Tom Hanks.jpg")
    (tmp_path / "LEIA-ME.md").write_text("instruções")
    (tmp_path / ".DS_Store").write_bytes(b"lixo do mac")
    assert build.local_actors(tmp_path) == ["Tom Hanks"]


def test_local_actors_is_empty_for_a_missing_folder(tmp_path):
    assert build.local_actors(tmp_path / "nao-existe") == []


def test_local_photos_finds_a_subfolder(tmp_path):
    foto_vazia(tmp_path / "Tom Hanks" / "b.jpg")
    foto_vazia(tmp_path / "Tom Hanks" / "a.jpg")
    encontradas = build.local_photos(tmp_path, "Tom Hanks")
    assert [p.name for p in encontradas] == ["a.jpg", "b.jpg"]


def test_local_photos_finds_a_loose_file(tmp_path):
    foto_vazia(tmp_path / "Alice Braga.jpg")
    assert [p.name for p in build.local_photos(tmp_path, "Alice Braga")] == ["Alice Braga.jpg"]


def test_local_photos_accepts_the_slug_as_folder_name(tmp_path):
    foto_vazia(tmp_path / "tais-araujo" / "1.jpg")
    assert len(build.local_photos(tmp_path, "Taís Araújo")) == 1


def test_local_photos_ignores_files_that_are_not_images(tmp_path):
    foto_vazia(tmp_path / "Tom Hanks" / "1.jpg")
    (tmp_path / "Tom Hanks" / "anotacoes.txt").write_text("nao é foto")
    assert len(build.local_photos(tmp_path, "Tom Hanks")) == 1


def test_photos_dir_build_uses_the_folder_names(base, photo, monkeypatch):
    """O caminho completo do modo pasta: nomes vêm do disco, não de actors.py."""
    pytest.importorskip("torch")
    from app import face

    try:
        face.warmup()
    except Exception as exc:
        pytest.skip(f"modelo indisponível: {exc}")

    fotos = base / "fotos"
    (fotos / "Pessoa Um").mkdir(parents=True)
    (fotos / "Pessoa Um" / "1.jpg").write_bytes(photo)
    (fotos / "Pessoa Dois.jpg").write_bytes(photo)

    monkeypatch.setattr(build, "ACTORS", ["Alguem Que Nao Esta Na Pasta"])
    monkeypatch.setattr("sys.argv", ["build_actors", "--photos-dir", str(fotos)])

    assert build.main() == 0
    assert set(build.load_existing()) == {"Pessoa Um", "Pessoa Dois"}


def test_photos_dir_grows_the_database_little_by_little(base, photo, monkeypatch):
    pytest.importorskip("torch")
    from app import face

    try:
        face.warmup()
    except Exception as exc:
        pytest.skip(f"modelo indisponível: {exc}")

    fotos = base / "fotos"
    fotos.mkdir()
    (fotos / "Primeira Pessoa.jpg").write_bytes(photo)
    monkeypatch.setattr("sys.argv", ["build_actors", "--photos-dir", str(fotos)])
    assert build.main() == 0

    # Semana seguinte: mais uma foto na pasta, e a anterior tem que continuar lá.
    (fotos / "Segunda Pessoa.jpg").write_bytes(photo)
    monkeypatch.setattr("sys.argv", ["build_actors", "--photos-dir", str(fotos), "--merge"])
    assert build.main() == 0

    assert set(build.load_existing()) == {"Primeira Pessoa", "Segunda Pessoa"}


def test_photos_dir_complains_when_the_folder_is_empty(base, monkeypatch, caplog):
    vazia = base / "fotos"
    vazia.mkdir()
    monkeypatch.setattr("sys.argv", ["build_actors", "--photos-dir", str(vazia)])
    monkeypatch.setattr(build.face, "warmup", lambda: None)

    with caplog.at_level("ERROR", logger="build_actors"):
        assert build.main() == 1
    assert "Nenhuma foto" in caplog.text


# --- progresso do download ------------------------------------------------


def test_fetch_all_reports_progress_for_each_person(base, monkeypatch, caplog):
    """Sem sinal de vida, meia hora de download parece um travamento."""
    monkeypatch.setattr(build, "photo_urls", lambda a, n: ["https://e.com/x.jpg"])
    monkeypatch.setattr(build, "download", lambda a, urls: [Path("x.jpg")])

    with caplog.at_level("INFO", logger="build_actors"):
        build.fetch_all(["Ator Um", "Ator Dois", "Ator Tres"], per_actor=4, workers=1, rate=0.0)

    texto = caplog.text
    assert "estimativa" in texto
    for nome in ("Ator Um", "Ator Dois", "Ator Tres"):
        assert nome in texto
    assert "[  3/3]" in texto  # o contador chega ao fim
    assert "faltam" in texto


def test_fetch_all_estimate_grows_with_the_rate(base, monkeypatch, caplog):
    monkeypatch.setattr(build, "photo_urls", lambda a, n: [])
    monkeypatch.setattr(build, "download", lambda a, urls: [])

    def estimativa(rate):
        caplog.clear()
        with caplog.at_level("INFO", logger="build_actors"):
            build.fetch_all(["A"] * 60, per_actor=4, workers=1, rate=rate)
        linha = next(l for l in caplog.text.splitlines() if "estimativa" in l)
        return float(linha.split("estimativa: ")[1].split(" min")[0])

    assert estimativa(2.0) > estimativa(1.0) > 0
