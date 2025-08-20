#!/usr/bin/env bash
set -euo pipefail

# Ensure we run at repo root regardless of caller CWD
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Ensure pip-tools is available
python -m pip install --upgrade pip setuptools wheel pip-tools >/dev/null 2>&1 || true

# Compile a fully hashed lock file from requirements.in
pip-compile --quiet --generate-hashes --output-file requirements.lock.txt requirements.in

echo "Generated requirements.lock.txt at $ROOT_DIR/requirements.lock.txt"
