#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/SteveKGYang/MentalLLaMA.git"

def main():
    target = Path("data/mentallama")
    if target.exists() and any(target.iterdir()):
        print(f"Target {target} not empty; skipping clone.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git","clone",REPO,str(target)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git clone failed: {e}")
        sys.exit(1)
    print(f"Cloned into {target}")

if __name__ == "__main__":
    main()
