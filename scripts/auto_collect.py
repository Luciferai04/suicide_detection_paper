#!/usr/bin/env python3
"""
Auto Collector: waits for SVM and BiLSTM completion, then runs results collection.
"""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

SESSION = "20250819_231757"
LOGS = Path("logs")
RESULTS = Path("results")

SVM_PID = LOGS / f"svm_{SESSION}.pid"
BILSTM_PID = LOGS / f"bilstm_{SESSION}.pid"
BERT_PID = LOGS / f"bert_{SESSION}.pid"


def is_alive(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_completion():
    print(f"[auto-collect] Monitoring SVM and BiLSTM completion for session {SESSION}...")
    while True:
        svm_done = not is_alive(SVM_PID)
        bilstm_done = not is_alive(BILSTM_PID)
        bert_done = not is_alive(BERT_PID)
        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{ts}] Status: SVM done={svm_done}, BiLSTM done={bilstm_done}, BERT done={bert_done}"
        )
        if svm_done and bilstm_done:
            break
        time.sleep(60)


def collect_results():
    print("[auto-collect] Running comprehensive results collection...")
    proc = subprocess.run(["python3", "collect_results.py"], capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)


def main():
    wait_for_completion()
    collect_results()
    print("[auto-collect] Done. Reports and JSON saved in results/")


if __name__ == "__main__":
    main()
