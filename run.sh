#!/usr/bin/env bash
# Sobe o servidor local em http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -r requirements.txt
fi

if [ ! -f data/actors.npz ]; then
  echo "Base de atores ausente — gerando (pode levar alguns minutos)..."
  ./.venv/bin/python -m scripts.build_actors
fi

exec ./.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port "${PORT:-8000}"
