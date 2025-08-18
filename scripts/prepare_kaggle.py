#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from suicide_detection.data_processing import Anonymizer
from suicide_detection.data_processing import reddit_clean


def main():
    ap = argparse.ArgumentParser(description="Prepare Kaggle SuicideWatch dataset: anonymize, clean, split")
    ap.add_argument("--infile", default="data/kaggle/Suicide_Detection.csv")
    ap.add_argument("--outdir", default="data/kaggle/splits")
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    # Expect columns: text, class
    if 'text' not in df.columns:
        raise ValueError("Expected 'text' column in Kaggle CSV")
    label_col = 'class' if 'class' in df.columns else 'label'
    df[label_col] = df[label_col].astype(str).str.lower().map({'suicide':1,'suicidal':1,'1':1,'non-suicide':0,'non-suicidal':0,'0':0}).fillna(df[label_col])
    df['label'] = df[label_col].astype(int)

    # Anonymize + clean
    anon = Anonymizer()
    df['text'] = df['text'].astype(str).map(anon.transform).map(reddit_clean)

    # Split 70/15/15 stratified
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
