#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_VIZ = ROOT / "results/visualizations"
OUT_TAB = ROOT / "results/comparison_tables"


def main():
    rows = []
    for model in ["svm", "bilstm", "bert"]:
        for split in ["val", "test"]:
            metrics_path = ROOT / f"results/model_outputs/{model}/{model}_{split}_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                rows.append({"model": model, "split": split, **m})
    if rows:
        df = pd.DataFrame(rows)
        OUT_TAB.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_TAB / "model_comparison.csv", index=False)
        print(f"Wrote {OUT_TAB / 'model_comparison.csv'}")
    else:
        print("No metrics found. Run training scripts first.")

if __name__ == "__main__":
    main()
