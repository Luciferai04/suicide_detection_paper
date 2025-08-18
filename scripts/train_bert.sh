#!/usr/bin/env bash
set -euo pipefail

# Fine-tune BERT/RoBERTa
ENV=${1:-.venv}
. "$ENV"/bin/activate
python -m suicide_detection.training.train \
  --model bert \
  --train_path data/splits/train.csv \
  --val_path data/splits/val.csv \
  --test_path data/splits/test.csv \
  --output_dir results/model_outputs/bert \
  --config configs/bert.yaml \
  --default_config configs/default.yaml

