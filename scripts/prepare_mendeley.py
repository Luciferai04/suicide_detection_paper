#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
# Ensure local package import without relying on editable install
sys.path.append(str((Path(__file__).resolve().parents[1] / 'src')))
import json
import pandas as pd
from suicide_detection.data_processing import Anonymizer
from suicide_detection.data_processing import reddit_clean


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in {'.json', '.jsonl'}:
        try:
            return pd.read_json(path, lines=path.suffix.lower()=='.jsonl')
        except ValueError:
            # try standard json list
            data = json.loads(path.read_text())
            return pd.DataFrame(data)
    return pd.DataFrame()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower():c for c in df.columns}
    text_col = cols.get('text') or cols.get('post') or cols.get('body')
    label_col = cols.get('label') or cols.get('class') or cols.get('suicidal')
    if not text_col or not label_col:
        raise ValueError("Could not find text/label columns in Mendeley file")
    out = pd.DataFrame({
        'text': df[text_col].astype(str),
        'label': df[label_col].astype(str).str.lower().map({'suicidal':1,'1':1,'non-suicidal':0,'0':0}).fillna(df[label_col]).astype(int)
    })
    return out


def main():
    ap = argparse.ArgumentParser(description="Prepare Mendeley datasets: merge, anonymize, clean, split")
    ap.add_argument("--indir", default="data/mendeley")
    ap.add_argument("--outdir", default="data/mendeley/splits")
    ap.add_argument("--group_cols", nargs="*", default=[], help="Optional quasi-identifier columns to check k-anonymity (if present)")
    ap.add_argument("--k_anonymity_k", type=int, default=5)
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in indir.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.csv','.json','.jsonl'}:
            try:
                df = load_any(p)
                if not df.empty:
                    frames.append(normalize(df))
            except Exception:
                continue
    if not frames:
        raise SystemExit("No usable Mendeley files found. Ensure downloads are placed under data/mendeley.")

    df = pd.concat(frames, ignore_index=True).dropna().drop_duplicates(subset=['text'])
    # Anonymize + clean
    anon = Anonymizer()
    df['text'] = df['text'].map(anon.transform).map(reddit_clean)

    # Optional k-anonymity report if group columns exist
    if args.group_cols:
        try:
            from suicide_detection.ethics.privacy import k_anonymity_report, write_privacy_report
            rep = k_anonymity_report(df, args.group_cols, k=args.k_anonymity_k)
            write_privacy_report(rep, Path("results/privacy") / "mendeley_k_anon.json")
        except Exception:
            pass

    # Split without sklearn (stratified)
    import numpy as np
    rng = np.random.default_rng(42)
    X = df['text'].values
    y = df['label'].values
    idx = np.arange(len(y))
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
