#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from suicide_detection.bias_audit.audit import demographic_group_metrics


def main():
    ap = argparse.ArgumentParser(description="Compute subgroup metrics for bias auditing")
    ap.add_argument("--labels", required=True, help="CSV with columns: id(optional), label")
    ap.add_argument(
        "--probs", required=True, help="CSV with columns: id(optional), prob (positive class)"
    )
    ap.add_argument("--groups", required=True, help="CSV with columns: id(optional), group")
    ap.add_argument("--out", required=True, help="Output markdown report path")
    args = ap.parse_args()

    df_y = pd.read_csv(args.labels)
    df_p = pd.read_csv(args.probs)
    df_g = pd.read_csv(args.groups)

    # Align by id if present
    if "id" in df_y.columns and "id" in df_p.columns and "id" in df_g.columns:
        df = df_y.merge(df_p, on="id").merge(df_g, on="id")
        y_true = df["label"].values.astype(int)
        y_prob = df["prob"].values.astype(float)
        groups = df["group"].values
    else:
        # assume same order
        y_true = df_y["label"].values.astype(int)
        y_prob = df_p["prob"].values.astype(float)
        groups = df_g["group"].values

    metrics = demographic_group_metrics(y_true, y_prob, groups)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Bias Audit Report\n\n")
        if not metrics:
            f.write("No groups provided or empty groups. No audit performed.\n")
        else:
            for g, vals in metrics.items():
                f.write(f"## Group: {g}\n")
                for k, v in vals.items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
