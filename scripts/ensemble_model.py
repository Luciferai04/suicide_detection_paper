#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import json

from suicide_detection.evaluation.metrics import compute_metrics


def load_probs(base: Path, model: str, split: str):
    p = base / f"{model}_{split}_probs.npy"
    y = base / f"{model}_{split}_y.npy"
    if not p.exists() or not y.exists():
        raise FileNotFoundError(f"Missing files for {model} {split}: {p} or {y}")
    return np.load(p), np.load(y)


def main():
    ap = argparse.ArgumentParser(description="Combine model probabilities into an ensemble and evaluate")
    ap.add_argument("--outputs_dir", default="results/model_outputs", help="Directory containing per-model outputs")
    ap.add_argument("--dataset", required=True, help="Dataset name suffix used in outputs, e.g., kaggle")
    ap.add_argument("--models", nargs="*", default=["svm","bilstm","bert"], help="Models to ensemble")
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="Optional weights for models (same length as models)")
    ap.add_argument("--split", default="test", choices=["val","test"], help="Which split to evaluate for the ensemble")
    ap.add_argument("--out", default=None, help="Output JSON file for ensemble metrics (auto if not set)")
    args = ap.parse_args()

    base = Path(args.outputs_dir)
    probs = []
    y_true = None
    for i, m in enumerate(args.models):
        model_dir = base / f"{m}_{args.dataset}"
        p, y = load_probs(model_dir, m, args.split)
        probs.append(p)
        if y_true is None:
            y_true = y
        else:
            if len(y) != len(y_true):
                raise ValueError("All models must have predictions for the same number of samples")
    probs = np.stack(probs, axis=0)

    if args.weights is None:
        w = np.ones(probs.shape[0]) / probs.shape[0]
    else:
        w = np.array(args.weights, dtype=float)
        if w.shape[0] != probs.shape[0]:
            raise ValueError("weights must match number of models")
        w = w / (w.sum() + 1e-8)

    ens = (w[:, None] * probs).sum(axis=0)
    res = compute_metrics(y_true, ens)

    out_dir = base / f"ensemble_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = Path(args.out) if args.out else out_dir / f"ensemble_{args.split}_metrics.json"
    with open(out_fp, "w") as f:
        json.dump(res.__dict__, f, indent=2)
    print(f"Saved ensemble metrics to {out_fp}")


if __name__ == "__main__":
    main()

