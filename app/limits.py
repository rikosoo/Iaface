"""Controle de carga e de abuso para quando o app fica exposto na internet.

Em `localhost` nada disso importa: um usuário, uma análise por vez. Publicado,
o `/api/match` é o endpoint caro do sistema — cada chamada segura CPU e algumas
centenas de MB — e precisa de duas defesas diferentes:

- **Concorrência**: um teto de análises simultâneas. Sem isso, um pico de
  acessos abre uma análise por requisição até a memória acabar.
- **Ritmo por IP**: um teto de chamadas por minuto. Sem isso, um script simples
  ocupa a fila inteira sozinho.

Tudo é guardado em memória, de propósito: são contadores efêmeros, não dados de
usuário. Reiniciar o processo zera, e é isso mesmo que se espera.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app import config


class Semaforo:
    """Limita quantas análises rodam ao mesmo tempo.

    Quem chega com a casa cheia espera um pouco; se não abrir vaga, recebe 503
    com Retry-After em vez de ficar pendurado até o navegador desistir.
    """

    def __init__(self, maximo: int, espera: float):
        self._sem = asyncio.Semaphore(maximo)
        self._espera = espera
        self.maximo = maximo

    async def __aenter__(self):
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self._espera)
        except (asyncio.TimeoutError, TimeoutError):
            raise HTTPException(
                status_code=503,
                detail="Estamos com muita gente agora. Tente de novo em alguns segundos.",
                headers={"Retry-After": "5"},
            )
        return self

    async def __aexit__(self, *exc):
        self._sem.release()

    @property
    def livres(self) -> int:
        return self._sem._value


class LimitePorIP:
    """Janela deslizante de chamadas por IP."""

    def __init__(self, maximo: int, janela: float):
        self.maximo = maximo
        self.janela = janela
        self._historico: dict[str, deque[float]] = defaultdict(deque)

    def registrar(self, ip: str) -> None:
        agora = time.monotonic()
        marcas = self._historico[ip]

        while marcas and agora - marcas[0] > self.janela:
            marcas.popleft()

        if len(marcas) >= self.maximo:
            faltam = int(self.janela - (agora - marcas[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Muitas análises seguidas. Espere {faltam}s e tente de novo.",
                headers={"Retry-After": str(faltam)},
            )

        marcas.append(agora)
        self._limpar(agora)

    def _limpar(self, agora: float) -> None:
        """Descarta IPs inativos para o dicionário não crescer sem fim."""
        if len(self._historico) < 1000:
            return
        vazios = [ip for ip, m in self._historico.items() if not m or agora - m[-1] > self.janela]
        for ip in vazios:
            del self._historico[ip]


def ip_do_cliente(request: Request) -> str:
    """IP de quem chamou, considerando proxy reverso quando configurado.

    Só confiamos em X-Forwarded-For se o app foi explicitamente informado de
    que está atrás de um proxy — o cabeçalho é trivial de forjar, e confiar
    nele por padrão daria a qualquer um um jeito de furar o limite por IP.
    """
    if config.TRUST_PROXY:
        encaminhado = request.headers.get("x-forwarded-for")
        if encaminhado:
            return encaminhado.split(",")[0].strip()
    return request.client.host if request.client else "desconhecido"
