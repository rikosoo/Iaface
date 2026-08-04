"""Comparação do rosto com a base de atores e leitura do resultado."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from app import config


class ActorDatabaseMissing(Exception):
    """A base ainda não foi gerada por scripts/build_actors.py."""


@dataclass(frozen=True)
class Match:
    name: str
    thumb: str | None
    similarity: float
    percent: float
    label: str


@dataclass(frozen=True)
class ActorDatabase:
    names: list[str]
    thumbs: list[str]
    vectors: np.ndarray  # (n, 512), já normalizados

    def __len__(self) -> int:
        return len(self.names)

    def rank(self, vector: np.ndarray, top_n: int = config.TOP_N) -> list[Match]:
        """Atores mais próximos, do mais parecido para o menos.

        Os vetores dos dois lados são normalizados, então o produto escalar já
        é a similaridade de cosseno — não precisa dividir por norma nenhuma.
        """
        scores = self.vectors @ vector
        order = np.argsort(-scores)[:top_n]
        return [
            Match(
                name=self.names[i],
                thumb=f"/static/actors/{self.thumbs[i]}" if self.thumbs[i] else None,
                similarity=round(float(scores[i]), 4),
                percent=to_percent(float(scores[i])),
                label=describe(float(scores[i])),
            )
            for i in order
        ]


_db: ActorDatabase | None = None
_lock = threading.Lock()


def load() -> ActorDatabase:
    """Carrega a base do disco uma única vez, mesmo com pedidos simultâneos."""
    global _db
    if _db is None:
        with _lock:
            if _db is None:
                if not config.ACTORS_NPZ.exists():
                    raise ActorDatabaseMissing(str(config.ACTORS_NPZ))
                data = np.load(config.ACTORS_NPZ, allow_pickle=False)
                _db = ActorDatabase(
                    names=[str(n) for n in data["names"]],
                    thumbs=[str(t) for t in data["thumbs"]],
                    vectors=np.ascontiguousarray(data["vectors"], dtype=np.float32),
                )
    return _db


def to_percent(similarity: float) -> float:
    """Cosseno -> porcentagem legível.

    O cosseno útil vive entre ~0,10 e ~0,75; esticar essa faixa dá um número
    que corresponde melhor ao que a pessoa vê na tela. É escala de leitura,
    não probabilidade.
    """
    span = config.PERCENT_CEIL - config.PERCENT_FLOOR
    return round(float(np.clip((similarity - config.PERCENT_FLOOR) / span, 0, 1)) * 100, 1)


def describe(similarity: float) -> str:
    """Tradução do cosseno para uma frase curta."""
    for threshold, label in config.SIMILARITY_BANDS:
        if similarity >= threshold:
            return label
    return config.SIMILARITY_BANDS[-1][1]


def verdict(matches: list[Match]) -> str:
    """Frase que resume o ranking, considerando a distância entre 1º e 2º."""
    if not matches:
        return "Não encontrei ninguém para comparar."

    first = matches[0]
    if first.similarity < 0.32:
        return (
            f"Ninguém da base parece muito com você — o mais próximo foi "
            f"{first.name}, mas por pouca coisa."
        )

    gap = first.similarity - (matches[1].similarity if len(matches) > 1 else 0.0)
    if gap >= 0.10:
        return f"{first.name} ficou bem à frente dos outros: é o seu par mais claro."
    if gap >= 0.04:
        return f"{first.name} lidera, com {matches[1].name} logo atrás."
    return f"Empate técnico entre {first.name} e {matches[1].name} — você tem um pouco dos dois."
