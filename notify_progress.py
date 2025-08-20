#!/usr/bin/env python3
"""
Background Training Progress Notifier
Runs in background and generates notification files for training progress
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime

def check_training_status():
    """Check current status and generate notification"""
    timestamp = "20250819_231757"
    models = ['svm', 'bilstm', 'bert']
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    for model in models:
        pid_file = f'logs/{model}_{timestamp}.pid'
        log_file = f'logs/{model}_{timestamp}.log'
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            # Check if process is alive
            try:
                os.kill(pid, 0)
                is_running = True
            except OSError:
                is_running = False
                
            # Get log size as progress indicator
            log_size = os.path.getsize(log_file) if os.path.exists(log_file) else 0
            
            status['models'][model] = {
                'pid': pid,
                'running': is_running,
                'log_size_bytes': log_size,
                'status': 'running' if is_running else 'completed'
            }
            
        except Exception as e:
            status['models'][model] = {
                'error': str(e),
                'status': 'error'
            }
    
    # Write status to notification file
    Path('notifications').mkdir(exist_ok=True)
    with open('notifications/training_status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    return status

def main():
    """Run background notification loop"""
    print("📱 Background notifier started")
    
    while True:
        try:
            status = check_training_status()
            running_models = sum(1 for m in status['models'].values() if m.get('status') == 'running')
            completed_models = sum(1 for m in status['models'].values() if m.get('status') == 'completed')
            
            if completed_models == 3:
                print("🎉 All training completed! Stopping notifier.")
                break
                
            # Wait 5 minutes between checks
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("🛑 Notifier stopped")
            break
        except Exception as e:
            print(f"❌ Notifier error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
