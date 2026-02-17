#!/usr/bin/env sh
set -eu

PORT="${API_PORT:-8000}"
WORKERS="${API_WORKERS:-2}"
APP_ENV="${APP_ENV:-dev}"

python3 validate_env.py

exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers "${WORKERS}"
