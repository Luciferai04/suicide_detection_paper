#!/usr/bin/env python3
from __future__ import annotations

"""Prepare MentaLLaMA data for binary suicide risk classification.

Ethics: This script only converts existing public data. It does NOT generate synthetic
suicide-related content. It anonymizes PII and records an audit log of the operation.

Usage:
  python scripts/prepare_mentallama.py \
    --repo data/mentallama \
    --out data/mentallama/splits

It searches JSON files under the repo for suicide/self-harm instructions and
attempts to map them into a classification dataset with columns: text,label.
Label mapping is conservative and keyword-based; manual review is recommended.
"""

import argparse
from pathlib import Path
import json
import re
import pandas as pd

# Local package
from suicide_detection.data_processing.anonymize import Anonymizer
from suicide_detection.data_processing.reddit_text import reddit_clean
from suicide_detection.ethics.audit_logging import AuditLogger

SUICIDE_PAT = re.compile(r"\b(suicide|self[- ]?harm|kill\s*myself|ending\s*my\s*life)\b", re.I)


def iter_json_files(repo: Path):
    for p in repo.rglob("*.json"):
        if p.is_file():
            yield p


def extract_examples(p: Path):
    """Return list of {text: str, label: int} from a MentalLLaMA-like JSON file.

    Heuristic mapping (subject to manual validation):
      - If instruction/input/output contains suicide/self-harm terms -> label=1
      - Else -> label=0
    """
    out = []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return out
    if isinstance(data, dict):
        data = [data]
    for item in data:
        fields = []
        for k in ("instruction", "input", "output", "text", "content"):
            v = item.get(k)
            if isinstance(v, str):
                fields.append(v)
        if not fields:
            continue
        blob = "\n\n".join(fields)
        label = 1 if SUICIDE_PAT.search(blob or "") else 0
        out.append({"text": blob, "label": label})
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert MentaLLaMA repo data into classification splits")
    ap.add_argument("--repo", default="data/mentallama")
    ap.add_argument("--out", default="data/mentallama/splits")
    ap.add_argument("--group_cols", nargs="*", default=[], help="Optional quasi-identifiers to check k-anonymity if present")
    ap.add_argument("--k_anonymity_k", type=int, default=5)
    args = ap.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = AuditLogger(Path("logs/orchestration_mentallama.log"))
    audit.log_event("mentallama_prepare_start", {"repo": str(repo), "out_dir": str(out_dir)})

    rows = []
    # Prefer likely subdirs in repo for JSON content
    candidates = [repo / "data", repo / "datasets", repo / "json", repo]
    seen = set()
    for base in candidates:
        if base.exists():
            for jf in base.rglob("*.json"):
                if jf.is_file() and jf not in seen:
                    rows.extend(extract_examples(jf))
                    seen.add(jf)
    if not rows:
        for jf in iter_json_files(repo):
            rows.extend(extract_examples(jf))
    if not rows:
        audit.log_event("mentallama_prepare_empty", {})
        print("No convertible examples found. Ensure the repo contains JSON data or point --repo to the JSON subfolder.")
        return

    df = pd.DataFrame(rows).dropna()
    # Anonymize + clean
    anon = Anonymizer()
    df["text"] = df["text"].astype(str).map(anon.transform).map(reddit_clean)
    # Deduplicate
    df = df.drop_duplicates(subset=["text"]) 

    # Optional k-anonymity report if group columns exist
    from pathlib import Path as _P
    try:
        if args.group_cols:
            from suicide_detection.ethics.privacy import k_anonymity_report, write_privacy_report
            rep = k_anonymity_report(df, args.group_cols, k=args.k_anonymity_k)
            write_privacy_report(rep, _P("results/privacy") / "mentallama_k_anon.json")
    except Exception:
        pass

    # Simple stratified split (70/15/15)
    import numpy as np
    rng = np.random.default_rng(42)
    X = df["text"].values
    y = df["label"].values.astype(int)
    idx = np.arange(len(y))
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(y):
        li = idx[y == label]
        rng.shuffle(li)
        n = len(li)
        n_test = max(1, int(0.15 * n))
        n_val = max(1, int(0.15 * (n - n_test)))
        test_i = li[:n_test]
        val_i = li[n_test:n_test + n_val]
        train_i = li[n_test + n_val:]
        test_idx.append(test_i); val_idx.append(val_i); train_idx.append(train_i)
    train_idx = np.concatenate(train_idx); val_idx = np.concatenate(val_idx); test_idx = np.concatenate(test_idx)

    def dump(idxs, name):
        pd.DataFrame({"text": X[idxs], "label": y[idxs]}).to_csv(out_dir / f"{name}.csv", index=False)
    dump(train_idx, "train"); dump(val_idx, "val"); dump(test_idx, "test")

    audit.log_event("mentallama_prepare_done", {"n": int(len(df))})
    print(f"Wrote splits to {out_dir}")


if __name__ == "__main__":
    main()

