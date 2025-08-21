#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Ensure local package import without relying on editable install
sys.path.append(str((Path(__file__).resolve().parents[1] / 'src')))
import csv
import zipfile

import pandas as pd

from suicide_detection.data_processing import Anonymizer, reddit_clean


def main():
    ap = argparse.ArgumentParser(description="Prepare Kaggle SuicideWatch dataset: anonymize, clean, split")
    ap.add_argument("--infile", default="data/kaggle/Suicide_Detection.csv")
    ap.add_argument("--outdir", default="data/kaggle/splits")
    ap.add_argument("--group_cols", nargs="*", default=[], help="Optional quasi-identifier columns to check k-anonymity (e.g., age gender)")
    ap.add_argument("--k_anonymity_k", type=int, default=5)
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # If the file has a ZIP signature, extract the inner CSV first
    if infile.exists():
        with open(infile, 'rb') as f:
            sig = f.read(4)
        if sig == b'PK\x03\x04':
            # It's actually a ZIP; extract CSV next to it
            with zipfile.ZipFile(infile, 'r') as zf:
                csv_members = [m for m in zf.namelist() if m.lower().endswith('.csv')]
                if not csv_members:
                    raise SystemExit('ZIP file does not contain a CSV')
                # Prefer Suicide_Detection.csv if present
                preferred = [m for m in csv_members if 'suicide' in m.lower()]
                member = preferred[0] if preferred else csv_members[0]
                extract_dir = infile.parent
                zf.extract(member, extract_dir)
                infile = extract_dir / member

    # Robust CSV loading with encoding fallback
    # Attempt robust parsing with multiple fallbacks for Kaggle CSV quirks
    try:
        df = pd.read_csv(infile)
    except Exception:
        try:
            df = pd.read_csv(infile, encoding="utf-8", engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(infile, encoding="latin-1", engine="python", on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL)
    # Identify text and label columns robustly
    cols_lower = {c.lower(): c for c in df.columns}
    text_candidates = [
        'text','post','body','content','clean_text','selftext','message','title_body'
    ]
    label_candidates = ['class','label','target','suicidal']
    text_col = None
    for t in text_candidates:
        if t in cols_lower:
            text_col = cols_lower[t]
            break
    if text_col is None:
        # Try common combination Title + Post
        if 'title' in cols_lower and ('post' in cols_lower or 'body' in cols_lower):
            t2 = cols_lower.get('post', cols_lower.get('body'))
            df['__text'] = df[cols_lower['title']].astype(str) + ' ' + df[t2].astype(str)
            text_col = '__text'
        else:
            raise ValueError(f"Could not find text column in Kaggle CSV. Columns: {list(df.columns)[:10]} ...")
    label_col = None
    for lbl in label_candidates:
        if lbl in cols_lower:
            label_col = cols_lower[lbl]
            break
    if label_col is None:
        raise ValueError("Could not find label/class column in Kaggle CSV")

    # Normalize labels to {0,1}
    df[label_col] = df[label_col].astype(str).str.lower().map({
        'suicide':1,'suicidal':1,'1':1,'yes':1,'true':1,
        'non-suicide':0,'non-suicidal':0,'0':0,'no':0,'false':0
    }).fillna(df[label_col])
    df['label'] = df[label_col].astype(int)
    df['text'] = df[text_col].astype(str)

    # Anonymize + clean
    anon = Anonymizer()
    df['text'] = df['text'].astype(str).map(anon.transform).map(reddit_clean)

    # Optional k-anonymity report if group columns exist
    if args.group_cols:
        try:
            from suicide_detection.ethics.privacy import k_anonymity_report, write_privacy_report
            rep = k_anonymity_report(df, args.group_cols, k=args.k_anonymity_k)
            write_privacy_report(rep, Path("results/privacy") / "kaggle_k_anon.json")
        except Exception:
            pass

    # Split 70/15/15 stratified without sklearn
    import numpy as np
    rng = np.random.default_rng(42)
    X = df['text'].values
    y = df['label'].values
    idx = np.arange(len(y))
    # Stratify by label
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(y):
        li = idx[y == label]
        rng.shuffle(li)
        n = len(li)
        n_test = max(1, int(0.15 * n))
        n_val = max(1, int(0.15 * (n - n_test)))
        test_i = li[:n_test]
        val_i = li[n_test:n_test+n_val]
        train_i = li[n_test+n_val:]
        test_idx.append(test_i)
        val_idx.append(val_i)
        train_idx.append(train_i)
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    def dump(idxs, name):
        pd.DataFrame({'text': X[idxs], 'label': y[idxs]}).to_csv(outdir/f"{name}.csv", index=False)
    dump(train_idx, 'train')
    dump(val_idx, 'val')
    dump(test_idx, 'test')
    print(f"Wrote splits to {outdir}")

if __name__ == '__main__':
    main()
