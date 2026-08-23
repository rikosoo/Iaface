"""Testes dos nomes de atores: desambiguação da Wikipédia e integridade da lista.

Um título como "Joe Cole" leva ao jogador de futebol, não ao ator — a Wikipédia
desambigua com "(actor)". Aqui está o que garante que esse sufixo funcione como
endereço sem vazar para a tela, e que a lista de atores continue sã.
"""

import pytest

from scripts import build_actors as build


# --- nomes com desambiguação da Wikipédia ---------------------------------


@pytest.mark.parametrize(
    "titulo,esperado",
    [
        ("Matt Smith (actor)", "Matt Smith"),
        ("Joe Cole (actor)", "Joe Cole"),
        ("Pedro Alonso (actor)", "Pedro Alonso"),
        ("Jane Doe (actress)", "Jane Doe"),
        ("Tom Hanks", "Tom Hanks"),  # sem sufixo, passa intacto
        # Parêntese que faz parte do nome artístico não pode ser removido.
        ("Robert Downey Jr.", "Robert Downey Jr."),
    ],
)
def test_display_name_drops_the_wikipedia_suffix(titulo, esperado):
    assert build.display_name(titulo) == esperado


def test_the_thumbnail_follows_the_display_name():
    """Senão a miniatura viraria matt-smith-actor.jpg e nunca casaria."""
    assert build.slugify(build.display_name("Matt Smith (actor)")) == "matt-smith"


def test_a_name_with_no_results_is_retried_with_the_actor_suffix(monkeypatch):
    consultados = []

    def busca(titulo):
        consultados.append(titulo)
        return ["https://e.com/foto.jpg"] if "(actor)" in titulo else []

    monkeypatch.setattr(build, "_main_photo", busca)
    monkeypatch.setattr(build, "_gallery", lambda t: [])

    assert build.photo_urls("Joe Cole", 4) == ["https://e.com/foto.jpg"]
    assert consultados == ["Joe Cole", "Joe Cole (actor)"]


def test_a_name_that_already_works_is_not_retried(monkeypatch):
    consultados = []

    def busca(titulo):
        consultados.append(titulo)
        return ["https://e.com/foto.jpg"]

    monkeypatch.setattr(build, "_main_photo", busca)
    monkeypatch.setattr(build, "_gallery", lambda t: [])

    build.photo_urls("Tom Hanks", 4)
    assert consultados == ["Tom Hanks"]  # uma consulta só


def test_a_disambiguated_name_is_not_retried_again(monkeypatch):
    """"Matt Smith (actor) (actor)" não existe — não vale gastar requisição."""
    consultados = []
    monkeypatch.setattr(build, "_main_photo", lambda t: consultados.append(t) or [])
    monkeypatch.setattr(build, "_gallery", lambda t: [])

    build.photo_urls("Matt Smith (actor)", 4)
    assert consultados == ["Matt Smith (actor)"]


def test_the_saved_entry_uses_the_clean_name(base, photo, monkeypatch, fake_wikipedia):
    pytest.importorskip("torch")
    from app import face

    try:
        face.warmup()
    except Exception as exc:
        pytest.skip(f"modelo indisponível: {exc}")

    monkeypatch.setattr(build, "API", fake_wikipedia)
    monkeypatch.setattr(build, "ACTORS", ["Fulano de Tal (actor)"])
    monkeypatch.setattr("sys.argv", ["build_actors", "--per-actor", "1"])

    assert build.main() == 0
    # Sem o sufixo na base, o card mostra o nome da pessoa e não o endereço.
    assert set(build.load_existing()) == {"Fulano de Tal"}


# --- integridade da lista de atores ---------------------------------------


def test_the_actor_list_has_no_duplicates():
    from data.actors import ACTORS as lista

    repetidos = {n for n in lista if lista.count(n) > 1}
    assert not repetidos, f"nomes repetidos gastariam download à toa: {repetidos}"


def test_every_production_cast_is_in_the_main_list():
    from data.actors import ACTORS as lista
    from data.actors import BY_PRODUCTION

    for producao, elenco in BY_PRODUCTION.items():
        faltando = [n for n in elenco if n not in lista]
        assert not faltando, f"{producao}: fora da lista principal: {faltando}"


def test_actor_names_look_like_wikipedia_titles():
    from data.actors import ACTORS as lista

    for nome in lista:
        assert nome == nome.strip()
        assert nome[0].isupper(), f"título da Wikipédia começa com maiúscula: {nome}"
        # Aspas tortas quebram a busca; a Wikipédia usa a reta.
        assert "’" not in nome, f"use apóstrofo reto em {nome!r}"


def test_every_actor_gets_a_distinct_thumbnail_filename():
    """Dois nomes com o mesmo slug sobrescreveriam a foto um do outro."""
    from data.actors import ACTORS as lista

    slugs = {}
    for nome in lista:
        slug = build.slugify(build.display_name(nome))
        assert slug not in slugs, f"{nome} e {slugs.get(slug)} colidem em {slug!r}"
        slugs[slug] = nome
