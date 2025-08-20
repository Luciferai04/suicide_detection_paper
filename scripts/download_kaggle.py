#!/usr/bin/env python3
import os
import sys
import zipfile
from pathlib import Path
import subprocess

DATASET="nikhileswarkomati/suicide-watch"


def main():
    target_dir = Path("data/kaggle")
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "suicide-watch.zip"

    # Verify Kaggle credentials
    kaggle_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home()/".kaggle"))
    cred = kaggle_dir/"kaggle.json"
    if not cred.exists():
        print("ERROR: Kaggle API credentials not found. Please place kaggle.json under ~/.kaggle or set KAGGLE_CONFIG_DIR.")
        sys.exit(1)

    # Download via Kaggle CLI
    cmd = ["kaggle","datasets","download","-d",DATASET,"-p",str(target_dir),"-f","Suicide_Detection.csv"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Kaggle download failed: {e}")
        sys.exit(1)

    # Handle direct CSV or ZIPs
    csvs = list(target_dir.glob("*.csv"))
    zips = list(target_dir.glob("*.zip"))
    if csvs:
        print(f"Downloaded CSVs: {[p.name for p in csvs]} in {target_dir}")
    elif zips:
        for zp in zips:
            with zipfile.ZipFile(zp, 'r') as zf:
                zf.extractall(target_dir)
        print(f"Extracted ZIPs to {target_dir}")
    else:
        print("ERROR: No CSV or ZIP file downloaded.")
        sys.exit(1)

if __name__ == "__main__":
    main()
