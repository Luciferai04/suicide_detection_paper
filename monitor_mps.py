#!/usr/bin/env python3
"""Monitor MPS training progress for all models."""

import os
import time
import json
from datetime import datetime
from pathlib import Path
import subprocess

def get_latest_log_lines(log_path, n=5):
    """Get last n lines from a log file."""
    try:
        if not os.path.exists(log_path):
            return None
        result = subprocess.run(['tail', '-n', str(n), log_path], 
                              capture_output=True, text=True)
        return result.stdout
    except:
        return None

def check_process_status(pid):
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def main():
    # Define log files to monitor
    log_files = {
        'BERT': 'logs/bert_mps_retry_20250820_021810.log',
        'BiLSTM': 'logs/bilstm_mps_retry_20250820_021813.log', 
        'SVM': 'logs/svm_mps_retry_20250820_021818.log'
    }
    
    # Process IDs from background jobs
    pids = {
        'BERT': 2202,
        'BiLSTM': 2257,
        'SVM': 2355
    }
    
    print("\n" + "="*80)
    print(f"MPS Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    for model, log_path in log_files.items():
        print(f"\n[{model}]")
        
        # Check process status
        pid = pids.get(model)
        if pid:
            if check_process_status(pid):
                print(f"  Status: ✅ Running (PID: {pid})")
            else:
                print(f"  Status: ❌ Stopped (PID: {pid})")
        
        # Get latest log lines
        latest = get_latest_log_lines(log_path, 3)
        if latest:
            print(f"  Latest activity:")
            for line in latest.strip().split('\n'):
                if line.strip():
                    # Truncate long lines
                    if len(line) > 100:
                        line = line[:97] + "..."
                    print(f"    {line}")
        else:
            print(f"  No log output yet")
    
    # Check for completed training
    print("\n" + "-"*80)
    print("Checking for completed models...")
    
    output_dirs = [
        'results/model_outputs/bert_mps',
        'results/model_outputs/bilstm_mps',
        'results/model_outputs/svm_mps'
    ]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            model_name = Path(output_dir).name.replace('_mps', '').upper()
            
            # Check for completion flag
            complete_flag = os.path.join(output_dir, 'training_complete.flag')
            metrics_file = os.path.join(output_dir, 'metrics.json')
            
            if os.path.exists(complete_flag):
                print(f"  ✅ {model_name}: Training complete!")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        print(f"     Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            elif os.path.exists(metrics_file):
                print(f"  🔄 {model_name}: In progress...")
            else:
                print(f"  ⏳ {model_name}: Starting...")

if __name__ == "__main__":
    main()
