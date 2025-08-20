#!/usr/bin/env bash
set -euo pipefail

if ! command -v pip-compile >/dev/null 2>&1; then
  echo "pip-compile not found. Install pip-tools first: pip install pip-tools" >&2
  exit 1
fi

pip-compile --generate-hashes --output-file requirements.txt requirements.in
echo "Updated requirements.txt with pinned versions and hashes."
