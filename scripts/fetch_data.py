#!/usr/bin/env python3
"""
Secure data fetch utility (whitelisted sources only).

This script does NOT scrape or download arbitrary content. It supports only
pre-approved dataset endpoints that you explicitly select. All downloads are
saved under data/raw and accompanied by a SHA-256 checksum file.

IMPORTANT: You are responsible for agreeing to each dataset's license/terms.
This tool avoids automatic acceptance; it prints the dataset URL for your
confirmation and can download only after --confirm is passed.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import argparse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"


WHITELIST: Dict[str, str] = {
    # Example placeholder links (user must replace with real, allowed URLs)
    "clpsych2015_sample": "https://example.com/CLPsych2015_sample.csv",
    "suicidewatch_sample": "https://example.com/SuicideWatch_sample.csv",
}


@dataclass
class FetchResult:
    path: Path
    sha256: str


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(WHITELIST.keys()))
    ap.add_argument("--filename", required=False, help="Override output filename")
    ap.add_argument("--confirm", action="store_true", help="Confirm you agree to the dataset terms")
    args = ap.parse_args()

    if not args.confirm:
        print("Refusing to download without --confirm. Please review dataset terms and re-run with --confirm.")
        print(f"URL: {WHITELIST[args.dataset]}")
        sys.exit(1)

    url = WHITELIST[args.dataset]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_name = args.filename or f"{args.dataset}.csv"
    out_path = RAW_DIR / out_name
    print(f"Downloading {url} -> {out_path}")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(2)

    digest = sha256sum(out_path)
    (out_path.with_suffix(out_path.suffix + ".sha256")).write_text(digest)
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
