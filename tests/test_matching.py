import numpy as np
import pytest

from app import config, matching


def unit(*values) -> np.ndarray:
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def db():
    return matching.ActorDatabase(
        names=["Ator A", "Ator B", "Ator C"],
        thumbs=["a.jpg", "", "c.jpg"],
        vectors=np.stack([unit(1, 0, 0), unit(0.8, 0.6, 0), unit(0, 0, 1)]),
    )


def test_ranks_by_cosine_similarity(db):
    matches = db.rank(unit(1, 0, 0))
    assert [m.name for m in matches] == ["Ator A", "Ator B", "Ator C"]
    assert matches[0].similarity == pytest.approx(1.0)
    assert matches[1].similarity == pytest.approx(0.8, abs=1e-4)
    assert matches[2].similarity == pytest.approx(0.0, abs=1e-6)


def test_respects_top_n(db):
    assert len(db.rank(unit(1, 0, 0), top_n=2)) == 2


def test_missing_thumb_becomes_none(db):
    by_name = {m.name: m for m in db.rank(unit(1, 0, 0))}
    assert by_name["Ator A"].thumb == "/static/actors/a.jpg"
    assert by_name["Ator B"].thumb is None


def test_percent_is_clamped_and_monotonic():
    assert matching.to_percent(-0.5) == 0.0
    assert matching.to_percent(1.0) == 100.0
    values = [matching.to_percent(s) for s in (0.2, 0.4, 0.6)]
    assert values == sorted(values)


def test_describe_covers_the_whole_range():
    assert matching.describe(0.95) == "muito parecido"
    assert matching.describe(0.45) == "parecido"
    assert matching.describe(0.0) == "pouca semelhança"
    assert matching.describe(-1.0) == "pouca semelhança"


def test_bands_are_sorted_from_high_to_low():
    thresholds = [t for t, _ in config.SIMILARITY_BANDS]
    assert thresholds == sorted(thresholds, reverse=True)


def make(name, similarity):
    return matching.Match(name, None, similarity, matching.to_percent(similarity), matching.describe(similarity))


def test_verdict_points_out_a_clear_winner():
    assert "bem à frente" in matching.verdict([make("A", 0.70), make("B", 0.45)])


def test_verdict_points_out_a_tie():
    assert "Empate" in matching.verdict([make("A", 0.55), make("B", 0.54)])


def test_verdict_is_honest_when_nothing_matches():
    assert "Ninguém" in matching.verdict([make("A", 0.20), make("B", 0.18)])


def test_verdict_survives_an_empty_ranking():
    assert matching.verdict([]) == "Não encontrei ninguém para comparar."


def test_missing_database_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTORS_NPZ", tmp_path / "nao_existe.npz")
    monkeypatch.setattr(matching, "_db", None)
    with pytest.raises(matching.ActorDatabaseMissing):
        matching.load()
