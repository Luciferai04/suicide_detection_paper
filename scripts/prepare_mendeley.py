#!/usr/bin/env python3
import argparse
from pathlib import Path
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

    # Split
    from sklearn.model_selection import train_test_split
    X = df['text'].values
    y = df['label'].values
    X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=0.15,stratify=y,random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval,test_size=0.1765,stratify=y_trainval,random_state=42)
    def dump(x,y,name):
        pd.DataFrame({'text':x,'label':y}).to_csv(outdir/f"{name}.csv", index=False)
    dump(X_train,y_train,'train')
    dump(X_val,y_val,'val')
    dump(X_test,y_test,'test')
    print(f"Wrote splits to {outdir}")

if __name__ == '__main__':
    main()
