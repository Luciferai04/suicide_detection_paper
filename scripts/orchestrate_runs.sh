#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

mkdir -p logs results/model_outputs/svm results/model_outputs/bilstm results/model_outputs/bert

log() { printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

ts() { date +%Y%m%d_%H%M%S; }

wait_for_pid() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" || true)"
    if [[ -n "${pid}" ]]; then
      log "Waiting for PID $pid to finish ..."
      while kill -0 "$pid" 2>/dev/null; do sleep 15; done
      log "PID $pid finished."
    fi
  fi
}

copy_svm_artifacts() {
  local src_dir="results/svm_kaggle"
  local dst_dir="results/model_outputs/svm"
  if [[ -d "$src_dir" ]]; then
    log "Copying SVM artifacts from $src_dir to $dst_dir"
    mkdir -p "$dst_dir"
    # Copy metrics and plots if present
    find "$src_dir" -type f \( -name 'svm_*_metrics.json' -o -name 'svm_*_fairness.json' -o -name 'svm_*.*png' -o -name 'svm_*.*jpg' -o -name 'svm_feature_importance.csv' \) -exec cp {} "$dst_dir" \; || true
  else
    log "SVM source dir $src_dir not found; skipping copy."
  fi
}

run_bilstm() {
  local logf="logs/bilstm_kaggle_$(ts).log"
  log "Starting BiLSTM training (logs: $logf)"
  python -m suicide_detection.training.train \
    --model bilstm \
    --train_path data/kaggle/splits/train.csv \
    --val_path data/kaggle/splits/val.csv \
    --test_path data/kaggle/splits/test.csv \
    --output_dir results/model_outputs/bilstm \
    >"$logf" 2>&1
  log "BiLSTM training finished."
}

run_bert() {
  local logf="logs/bert_kaggle_$(ts).log"
  log "Starting BERT training (logs: $logf)"
  python -m suicide_detection.training.train \
    --model bert \
    --train_path data/kaggle/splits/train.csv \
    --val_path data/kaggle/splits/val.csv \
    --test_path data/kaggle/splits/test.csv \
    --output_dir results/model_outputs/bert \
    >"$logf" 2>&1
  log "BERT training finished."
}

compare_models() {
  log "Generating comparison table"
  python scripts/compare_models.py || true
}

main() {
  # 1) Wait for existing SVM background run (if any)
  wait_for_pid "logs/svm_spark.pid"
  # 2) Copy SVM artifacts to the standard model_outputs path
  copy_svm_artifacts
  # 3) Run BiLSTM then BERT sequentially
  run_bilstm
  run_bert
  # 4) Compare
  compare_models
  log "All runs completed."
}

main "$@"
