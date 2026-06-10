#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/nemotron-asr}"

mkdir -p "$APP_DIR"
cd "$APP_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but not installed." >&2
  exit 1
fi

docker compose build
docker compose up -d
docker compose ps
