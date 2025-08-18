#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

from suicide_detection.data_processing.load import load_dataset_secure


def main():
    ap = argparse.ArgumentParser(description="Anonymize and validate dataset, then write to processed path")
    ap.add_argument("--input", required=True, help="Path to input CSV/TSV with columns text,label")
    ap.add_argument("--output", required=True, help="Path to write processed CSV")
    args = ap.parse_args()

    df = load_dataset_secure(Path(args.input), anonymize=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote processed dataset to {out}")


if __name__ == "__main__":
    main()

