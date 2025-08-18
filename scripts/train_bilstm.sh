#!/usr/bin/env bash
set -euo pipefail

# Train BiLSTM with Attention
ENV=${1:-.venv}
. "$ENV"/bin/activate
python -m suicide_detection.training.train \
  --model bilstm \
  --train_path data/splits/train.csv \
  --val_path data/splits/val.csv \
  --test_path data/splits/test.csv \
  --output_dir results/model_outputs/bilstm \
  --config configs/bilstm.yaml \
  --default_config configs/default.yaml
