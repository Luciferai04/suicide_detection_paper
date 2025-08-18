#!/usr/bin/env bash
set -euo pipefail

# Train SVM baseline with TF-IDF features
ENV=${1:-.venv}
. "$ENV"/bin/activate
python -m suicide_detection.training.train \
  --model svm \
  --train_path data/splits/train.csv \
  --val_path data/splits/val.csv \
  --test_path data/splits/test.csv \
  --output_dir results/model_outputs/svm \
  --config configs/svm.yaml \
  --default_config configs/default.yaml

