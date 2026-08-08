"""Testes do modo público — o que muda quando o app sai do localhost.

Aqui a inferência é substituída por um dublê: o que está sob teste é a camada
de proteção (consentimento, fila, limite por IP, CORS), não o modelo.
"""

import asyncio
import time

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import config, limits, matching


@pytest.fixture
def app_publico(tmp_path, monkeypatch):
    """Sobe o app em modo público, com a análise trocada por um dublê rápido."""
    monkeypatch.setattr(config, "PUBLIC", True)

    npz = tmp_path / "actors.npz"
    vetor = np.eye(1, 512, dtype=np.float32)
    np.savez(npz, names=np.array(["Ator"]), thumbs=np.array([""]), vectors=vetor)
    monkeypatch.setattr(config, "ACTORS_NPZ", npz)
    monkeypatch.setattr(matching, "_db", None)

    from app import server

    monkeypatch.setattr(server, "config", config)
    monkeypatch.setattr(server.face, "warmup", lambda: None)
    monkeypatch.setattr(
        server,
        "_analisar",
        lambda raw, db: server.MatchResponse(
            matches=[],
            verdict="ok",
            quality=server.QualityOut(
                face_pixels=1, sharpness=1, brightness=1, detection_prob=1, warnings=[]
            ),
            face_crop="data:image/jpeg;base64,x",
            style=None,
        ),
    )
    monkeypatch.setattr(server, "limite_ip", limits.LimitePorIP(config.RATE_LIMIT, config.RATE_WINDOW))

    with TestClient(server.app) as client:
        yield client

    matching._db = None


def enviar(client, consentimento=True):
    headers = {"Content-Type": "image/jpeg"}
    if consentimento:
        headers["X-Iaface-Consent"] = "granted"
    return client.post("/api/match", content=b"bytes de imagem", headers=headers)


# --- Consentimento --------------------------------------------------------


def test_public_mode_refuses_analysis_without_consent(app_publico):
    resposta = enviar(app_publico, consentimento=False)
    assert resposta.status_code == 403
    assert "autorizar" in resposta.json()["detail"]


def test_public_mode_accepts_with_consent(app_publico):
    assert enviar(app_publico).status_code == 200


def test_a_forged_consent_value_is_not_accepted(app_publico):
    resposta = app_publico.post(
        "/api/match",
        content=b"x",
        headers={"Content-Type": "image/jpeg", "X-Iaface-Consent": "talvez"},
    )
    assert resposta.status_code == 403


def test_status_tells_the_front_that_consent_is_required(app_publico):
    assert app_publico.get("/api/status").json()["public"] is True


def test_localhost_mode_does_not_ask_for_consent(tmp_path, monkeypatch):
    """Rodando na própria máquina, o portão de consentimento sai da frente."""
    monkeypatch.setattr(config, "PUBLIC", False)

    npz = tmp_path / "actors.npz"
    np.savez(
        npz,
        names=np.array(["Ator"]),
        thumbs=np.array([""]),
        vectors=np.eye(1, 512, dtype=np.float32),
    )
    monkeypatch.setattr(config, "ACTORS_NPZ", npz)
    monkeypatch.setattr(matching, "_db", None)

    from app import server

    monkeypatch.setattr(server.face, "warmup", lambda: None)
    monkeypatch.setattr(
        server,
        "_analisar",
        lambda raw, db: server.MatchResponse(
            matches=[],
            verdict="ok",
            quality=server.QualityOut(
                face_pixels=1, sharpness=1, brightness=1, detection_prob=1, warnings=[]
            ),
            face_crop="x",
            style=None,
        ),
    )

    with TestClient(server.app) as client:
        assert client.get("/api/status").json()["public"] is False
        assert enviar(client, consentimento=False).status_code == 200

    matching._db = None


# --- Limite por IP --------------------------------------------------------


def test_the_rate_limit_lets_the_allowance_through():
    limite = limits.LimitePorIP(maximo=3, janela=60)
    for _ in range(3):
        limite.registrar("1.2.3.4")


def test_the_rate_limit_blocks_the_next_one():
    limite = limits.LimitePorIP(maximo=2, janela=60)
    limite.registrar("1.2.3.4")
    limite.registrar("1.2.3.4")
    with pytest.raises(HTTPException) as erro:
        limite.registrar("1.2.3.4")
    assert erro.value.status_code == 429
    assert "Retry-After" in erro.value.headers


def test_the_rate_limit_is_per_ip():
    limite = limits.LimitePorIP(maximo=1, janela=60)
    limite.registrar("1.1.1.1")
    limite.registrar("2.2.2.2")  # vizinho não paga pelo excesso do outro


def test_the_window_slides():
    limite = limits.LimitePorIP(maximo=1, janela=0.15)
    limite.registrar("1.2.3.4")
    time.sleep(0.2)
    limite.registrar("1.2.3.4")  # já saiu da janela


def test_inactive_ips_are_forgotten_so_memory_does_not_grow():
    limite = limits.LimitePorIP(maximo=5, janela=0.05)
    for i in range(1100):
        limite.registrar(f"10.0.{i // 256}.{i % 256}")
    time.sleep(0.1)
    limite.registrar("192.168.0.1")
    assert len(limite._historico) < 1100


def test_a_real_request_hits_the_rate_limit(app_publico, monkeypatch):
    from app import server

    monkeypatch.setattr(server, "limite_ip", limits.LimitePorIP(maximo=2, janela=60))
    assert enviar(app_publico).status_code == 200
    assert enviar(app_publico).status_code == 200
    resposta = enviar(app_publico)
    assert resposta.status_code == 429
    assert resposta.headers["retry-after"]


# --- IP do cliente --------------------------------------------------------


class PedidoFalso:
    def __init__(self, headers, host="9.9.9.9"):
        self.headers = headers
        self.client = type("C", (), {"host": host})()


def test_forwarded_header_is_ignored_without_a_proxy(monkeypatch):
    """Confiar em X-Forwarded-For por padrão daria a qualquer um um IP novo."""
    monkeypatch.setattr(config, "TRUST_PROXY", False)
    pedido = PedidoFalso({"x-forwarded-for": "1.2.3.4"})
    assert limits.ip_do_cliente(pedido) == "9.9.9.9"


def test_forwarded_header_is_used_behind_a_proxy(monkeypatch):
    monkeypatch.setattr(config, "TRUST_PROXY", True)
    pedido = PedidoFalso({"x-forwarded-for": "1.2.3.4, 10.0.0.1"})
    assert limits.ip_do_cliente(pedido) == "1.2.3.4"


# --- Fila de concorrência -------------------------------------------------


def test_the_queue_caps_simultaneous_analyses():
    async def cenario():
        fila = limits.Semaforo(maximo=2, espera=5)
        async with fila:
            async with fila:
                assert fila.livres == 0
                # A terceira teria que esperar; confirmamos que ela bloqueia.
                with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                    await asyncio.wait_for(fila.__aenter__(), timeout=0.1)

    asyncio.run(cenario())


def test_a_full_queue_answers_503_instead_of_hanging():
    async def cenario():
        fila = limits.Semaforo(maximo=1, espera=0.1)
        async with fila:
            with pytest.raises(HTTPException) as erro:
                await fila.__aenter__()
        assert erro.value.status_code == 503
        assert erro.value.headers["Retry-After"]

    asyncio.run(cenario())


def test_the_queue_frees_up_after_the_analysis_ends():
    async def cenario():
        fila = limits.Semaforo(maximo=1, espera=1)
        async with fila:
            pass
        assert fila.livres == 1
        async with fila:  # não trava na segunda vez
            pass

    asyncio.run(cenario())


def test_status_reports_when_the_queue_is_full(app_publico, monkeypatch):
    from app import server

    monkeypatch.setattr(server, "fila", limits.Semaforo(maximo=1, espera=1))
    assert app_publico.get("/api/status").json()["busy"] is False


# --- CORS -----------------------------------------------------------------


def test_cors_is_closed_by_default():
    """Embutido em iframe a origem é a mesma; abrir CORS sem precisar é risco."""
    assert config.CORS_ORIGINS == []


def test_the_consent_header_is_allowed_when_cors_is_on(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CORS_ORIGINS", ["https://meusite.com"])

    npz = tmp_path / "actors.npz"
    np.savez(
        npz,
        names=np.array(["A"]),
        thumbs=np.array([""]),
        vectors=np.eye(1, 512, dtype=np.float32),
    )
    monkeypatch.setattr(config, "ACTORS_NPZ", npz)
    monkeypatch.setattr(matching, "_db", None)

    import importlib

    from app import server

    recarregado = importlib.reload(server)
    monkeypatch.setattr(recarregado.face, "warmup", lambda: None)

    with TestClient(recarregado.app) as client:
        resposta = client.options(
            "/api/match",
            headers={
                "Origin": "https://meusite.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-iaface-consent",
            },
        )
        assert resposta.status_code == 200
        assert resposta.headers["access-control-allow-origin"] == "https://meusite.com"

    monkeypatch.setattr(config, "CORS_ORIGINS", [])
    importlib.reload(server)
    matching._db = None
