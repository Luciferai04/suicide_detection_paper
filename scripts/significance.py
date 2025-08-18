#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

from suicide_detection.evaluation.stat_tests import mcnemar_test, bootstrap_ci


def main():
    ap = argparse.ArgumentParser(description="Run significance tests between two model outputs")
    ap.add_argument("--y_true", required=True, help="Path to y_true.npy or JSON list")
    ap.add_argument("--y_prob_a", required=True, help="Path to y_prob for model A (npy or JSON list)")
    ap.add_argument("--y_prob_b", required=True, help="Path to y_prob for model B (npy or JSON list)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    def load_vec(p):
        pth = Path(p)
        if pth.suffix == ".npy":
            return np.load(pth)
        else:
            return np.array(json.loads(Path(p).read_text()))

    y_true = load_vec(args.y_true).astype(int)
    y_prob_a = load_vec(args.y_prob_a).astype(float)
    y_prob_b = load_vec(args.y_prob_b).astype(float)

    y_pred_a = (y_prob_a >= args.threshold).astype(int)
    y_pred_b = (y_prob_b >= args.threshold).astype(int)

    stat, pval = mcnemar_test(y_true, y_pred_a, y_true, y_pred_b)

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    auc_a, ci_auc_a = bootstrap_ci(lambda y, p: roc_auc_score(y, p), y_true, y_prob_a)
    auc_b, ci_auc_b = bootstrap_ci(lambda y, p: roc_auc_score(y, p), y_true, y_prob_b)
    pr_a, ci_pr_a = bootstrap_ci(lambda y, p: average_precision_score(y, p), y_true, y_prob_a)
    pr_b, ci_pr_b = bootstrap_ci(lambda y, p: average_precision_score(y, p), y_true, y_prob_b)
    f1_a, ci_f1_a = bootstrap_ci(lambda y, p: f1_score(y, (p>=args.threshold).astype(int)), y_true, y_prob_a)
    f1_b, ci_f1_b = bootstrap_ci(lambda y, p: f1_score(y, (p>=args.threshold).astype(int)), y_true, y_prob_b)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "mcnemar": {"statistic": stat, "p_value": pval},
        "auc": {"A": {"point": auc_a, "ci": ci_auc_a}, "B": {"point": auc_b, "ci": ci_auc_b}},
        "pr_auc": {"A": {"point": pr_a, "ci": ci_pr_a}, "B": {"point": pr_b, "ci": ci_pr_b}},
        "f1": {"A": {"point": f1_a, "ci": ci_f1_a}, "B": {"point": f1_b, "ci": ci_f1_b}}
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
