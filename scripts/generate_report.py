#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"
MODEL_DIR = OUT_DIR / "model_outputs"
REPORT_FP = OUT_DIR / "final_report.html"


def load_metrics() -> pd.DataFrame:
    rows = []
    for model in ["svm", "bilstm", "bert"]:
        for split in ["val", "test"]:
            fp = MODEL_DIR / model / f"{model}_{split}_metrics.json"
            if fp.exists():
                try:
                    data = json.loads(fp.read_text())
                    if isinstance(data, dict) and "accuracy" in data:
                        rows.append({"model": model, "split": split, **data})
                    elif isinstance(data, dict) and "metrics" in data:
                        rows.append({"model": model, "split": split, **data["metrics"]})
                    else:
                        rows.append({"model": model, "split": split, **data})
                except Exception:
                    continue
    return pd.DataFrame(rows)


def _load_fairness() -> pd.DataFrame:
    rows = []
    for model in ["svm", "bilstm", "bert"]:
        for split in ["val", "test"]:
            fp = MODEL_DIR / model / f"{model}_{split}_fairness.json"
            if fp.exists():
                try:
                    obj = json.loads(fp.read_text())
                    for g, vals in obj.items():
                        rows.append({"model": model, "split": split, "group": g, **vals})
                except Exception:
                    continue
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "<html><body><h2>No metrics found. Run training scripts first.</h2></body></html>"
    # Limit to main metrics and include cost-weighted accuracy if present
    metric_cols = [
        c
        for c in [
            "model",
            "split",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "cost_weighted_accuracy",
        ]
        if c in df.columns
    ]
    parts = [
        "<html><head><title>Suicide Detection Research Report</title></head><body>",
        "<h1>Suicide Detection Research - Model Comparison</h1>",
        "<p><em>Sensitive research content. For IRB-approved research only.</em></p>",
        df[metric_cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Fairness (if provided)</h2>",
    ]
    fdf = _load_fairness()
    if not fdf.empty:
        parts.append(fdf.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        parts.append(
            "<p>No demographic group data was provided; fairness analysis not performed.</p>"
        )

    parts.append("<h2>Figures</h2>")
    # Attach available plots
    figs = list((OUT_DIR / "visualizations").glob("*.png")) + list((MODEL_DIR).glob("**/*.png"))
    for fig in figs:
        rel = fig.relative_to(ROOT)
        parts.append(
            f"<div><img src='{rel.as_posix()}' style='max-width:720px'><p>{rel.as_posix()}</p></div>"
        )
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    html = render_report(df)
    REPORT_FP.write_text(html)
    print(f"Wrote report to {REPORT_FP}")


if __name__ == "__main__":
    main()
