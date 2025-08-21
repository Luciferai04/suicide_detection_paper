#!/usr/bin/env python3
"""
Auto-analysis watcher: monitors model outputs and triggers analyses.
- Runs error_analysis.py for newly completed models (when metrics.json appears)
- Refreshes literature baseline comparisons
- Regenerates manuscript after all expected models are complete
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

EXPECTED_MODELS = {"bert_mps", "bilstm_mps", "svm_mps"}
OUTPUT_BASE = Path("results/model_outputs")
STATE_FILE = Path("run_state/auto_analyze_state.json")
LOG_FILE = Path("logs/auto_analyze.log")

LOG_FILE.parent.mkdir(exist_ok=True)
STATE_FILE.parent.mkdir(exist_ok=True)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    LOG_FILE.write_text(LOG_FILE.read_text() + line if LOG_FILE.exists() else line)
    print(line, end="")


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {"analyzed": []}
    return {"analyzed": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def has_metrics(model_dir: Path) -> bool:
    # Consider complete if a test metrics file exists
    return any((model_dir / f).exists() for f in [
        "bert_test_metrics.json", "bilstm_test_metrics.json", "svm_test_metrics.json", "metrics.json"
    ])


def run_cmd(cmd: list[str], timeout: int | None = None):
    try:
        subprocess.run(cmd, check=False, timeout=timeout)
    except Exception as e:
        log(f"Error running {' '.join(cmd)}: {e}")


def main():
    log("Auto-analysis watcher started")
    state = load_state()

    while True:
        try:
            if not OUTPUT_BASE.exists():
                time.sleep(20)
                continue

            completed = []
            for model_dir in OUTPUT_BASE.iterdir():
                if not model_dir.is_dir():
                    continue
                name = model_dir.name
                if name in state.get("analyzed", []):
                    continue
                if has_metrics(model_dir):
                    completed.append(model_dir)

            for model_dir in completed:
                name = model_dir.name
                log(f"Detected completion: {name}. Running error analysis...")
                run_cmd(["python", "scripts/error_analysis.py"])  # runs over all models
                state.setdefault("analyzed", []).append(name)
                save_state(state)

                # Refresh literature baselines after each model completion
                log("Refreshing literature baseline comparisons...")
                run_cmd(["python", "scripts/literature_baselines.py"])

            # If all expected models complete, generate manuscript
            completed_set = set(m for m in state.get("analyzed", []) if m in EXPECTED_MODELS)
            if EXPECTED_MODELS.issubset(completed_set):
                log("All expected models complete. Generating manuscript...")
                run_cmd(["python", "scripts/generate_manuscript.py"], timeout=300)
                log("Manuscript generation triggered.")
                # Optionally break, but keep watching for new runs

            time.sleep(30)
        except KeyboardInterrupt:
            log("Watcher stopped by user")
            break
        except Exception as e:
            log(f"Watcher error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()

