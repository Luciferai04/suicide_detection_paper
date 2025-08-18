#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import requests
import zipfile
from io import BytesIO

URLS = [
    "https://data.mendeley.com/public-files/datasets/zz8j36y24f/files?download=1",
    "https://data.mendeley.com/public-files/datasets/z8s6w86tr3/files?download=1",
]


def safe_download(url: str, out_dir: Path):
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR fetching {url}: {e}")
        return False
    # Try to detect zip
    ctype = resp.headers.get("Content-Type", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    if "zip" in ctype or url.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            zf.extractall(out_dir)
        print(f"Extracted ZIP from {url}")
        return True
    # Otherwise write to a file
    fname = out_dir / Path(url).name
    with open(fname, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {fname}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Download Mendeley suicide datasets")
    ap.add_argument("--out", default="data/mendeley", help="Output directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ok_any = False
    for url in URLS:
        ok = safe_download(url, out)
        ok_any = ok_any or ok
    if not ok_any:
        print("No files downloaded. You may need to download manually via the website due to licensing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
