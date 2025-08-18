#!/usr/bin/env bash
set -euo pipefail
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
