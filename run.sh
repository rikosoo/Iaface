#!/usr/bin/env bash
# Sobe o servidor local em http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

# O python.exe/bin dentro do .venv é o que conta: uma instalação interrompida
# deixa a pasta criada e vazia, e o erro só apareceria lá na frente.
if [ ! -x .venv/bin/python ]; then
  [ -d .venv ] && { echo "Ambiente virtual incompleto — refazendo..."; rm -rf .venv; }
  python3 -m venv .venv
fi

if [ ! -f .venv/.deps-ok ]; then
  echo "Instalando as dependências (cerca de 2 GB na primeira vez)..."
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
  # O facenet-pytorch pina torch <2.3.0, que não tem instalador para o Python
  # 3.13+. O código dele roda com o torch atual, então entra sem as deps.
  ./.venv/bin/python -m pip install --no-deps facenet-pytorch==2.6.0
  touch .venv/.deps-ok
fi

if [ ! -f data/actors.npz ]; then
  echo "Base de atores ausente — gerando (pode levar alguns minutos)..."
  ./.venv/bin/python -m scripts.build_actors
fi

exec ./.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port "${PORT:-8000}"
