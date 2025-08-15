#!/usr/bin/env sh
set -e

mkdir -p /app/models

python scripts/bootstrap_models.py

exec "$@"
