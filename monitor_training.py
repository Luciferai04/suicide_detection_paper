#!/usr/bin/env python3
"""
Training Monitor Script
Monitors SVM, BiLSTM, and BERT training progress and provides periodic updates.
"""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def check_process_alive(pid):
    """Check if process is still running"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_process_stats(pid):
    """Get CPU and memory usage for a process"""
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'pid,pcpu,pmem,time,args'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                stats = lines[1].strip().split(None, 4)
                return {
                    'pid': stats[0],
                    'cpu': stats[1],
                    'memory': stats[2], 
                    'time': stats[3],
                    'status': 'running'
                }
    except Exception:
        pass
    return {'status': 'not_running'}

def tail_log(logfile, lines=5):
    """Get last N lines from log file"""
    try:
        result = subprocess.run(
            ['tail', '-n', str(lines), logfile],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "No log output"

def check_output_dir(output_dir):
    """Check what artifacts are in the output directory"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return {"status": "not_created", "files": []}
    
    files = list(output_path.glob("*.json"))
    plots = list(output_path.glob("*.png"))
    
    return {
        "status": "created",
        "json_files": [f.name for f in files],
        "plots": [f.name for f in plots],
        "total_files": len(list(output_path.iterdir()))
    }

def extract_bert_progress(log_content):
    """Extract BERT training progress from log"""
    lines = log_content.split('\n')
    progress_line = None
    for line in reversed(lines):
        if '|' in line and '%' in line and 'it/s' in line:
            progress_line = line
            break
    
    if progress_line:
        try:
            # Extract percentage and current/total steps
            if '%|' in progress_line:
                parts = progress_line.split('%|')[0]
                percentage = parts.split()[-1] if parts else "0"
                return f"{percentage}% progress"
        except:
            pass
    
    return "Training in progress"

def extract_bilstm_progress(log_content):
    """Extract BiLSTM training progress from log"""
    if not log_content.strip():
        return "Starting up..."
    
    lines = log_content.split('\n')
    for line in reversed(lines):
        if 'epoch' in line.lower() or 'loss' in line.lower():
            return line.strip()
    
    return "Training in progress"

def extract_svm_progress(log_content):
    """Extract SVM training progress from log"""
    lines = log_content.split('\n')
    for line in reversed(lines):
        if 'fit' in line.lower() or 'best' in line.lower() or 'params' in line.lower():
            return line.strip()
    
    return "Grid search in progress"

def main():
    # Training configuration
    timestamp = "20250819_231757"
    models = {
        'svm': {
            'pid_file': f'logs/svm_{timestamp}.pid',
            'log_file': f'logs/svm_{timestamp}.log',
            'output_dir': f'results/model_outputs/svm_run_{timestamp}',
            'mlflow_run': 'ea2cfae5fd8f472a9283a73f8e09a530'
        },
        'bilstm': {
            'pid_file': f'logs/bilstm_{timestamp}.pid',
            'log_file': f'logs/bilstm_{timestamp}.log',
            'output_dir': f'results/model_outputs/bilstm_run_{timestamp}',
            'mlflow_run': 'd71bebb05d624f1f908e2dd54fed6971'
        },
        'bert': {
            'pid_file': f'logs/bert_{timestamp}.pid',
            'log_file': f'logs/bert_{timestamp}.log',
            'output_dir': f'results/model_outputs/bert_run_{timestamp}',
            'mlflow_run': 'a64209cd7b094e5eb0b267117afbce10'
        }
    }
    
    print(f"🔍 Training Monitor Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    completed_models = set()
    
    while len(completed_models) < 3:
        print(f"\n📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        for model_name, config in models.items():
            if model_name in completed_models:
                print(f"✅ {model_name.upper()}: COMPLETED")
                continue
                
            # Check if PID file exists
            if not os.path.exists(config['pid_file']):
                print(f"❌ {model_name.upper()}: PID file missing")
                continue
                
            # Read PID
            try:
                with open(config['pid_file'], 'r') as f:
                    pid = int(f.read().strip())
            except:
                print(f"❌ {model_name.upper()}: Could not read PID")
                continue
            
            # Check if process is alive
            if not check_process_alive(pid):
                print(f"🏁 {model_name.upper()}: Process completed (PID {pid})")
                completed_models.add(model_name)
                
                # Check final artifacts
                artifacts = check_output_dir(config['output_dir'])
                if artifacts['json_files']:
                    print(f"   📁 Artifacts: {', '.join(artifacts['json_files'])}")
                continue
            
            # Get process stats
            stats = get_process_stats(pid)
            if stats['status'] == 'running':
                cpu_usage = stats.get('cpu', '0.0')
                memory_usage = stats.get('memory', '0.0')
                runtime = stats.get('time', '0:00.00')
                
                # Get log progress
                log_content = tail_log(config['log_file'], 20)
                
                if model_name == 'svm':
                    progress = extract_svm_progress(log_content)
                elif model_name == 'bilstm':
                    progress = extract_bilstm_progress(log_content)
                elif model_name == 'bert':
                    progress = extract_bert_progress(log_content)
                
                print(f"🚀 {model_name.upper()}: Running (PID {pid}) | CPU: {cpu_usage}% | Memory: {memory_usage}% | Time: {runtime}")
                print(f"   📝 {progress}")
                
                # Check artifacts
                artifacts = check_output_dir(config['output_dir'])
                if artifacts.get('total_files', 0) > 0:
                    print(f"   📁 Files created: {artifacts['total_files']}")
            else:
                print(f"❓ {model_name.upper()}: Process status unknown")
        
        if len(completed_models) < 3:
            print(f"\n⏳ Monitoring... ({len(completed_models)}/3 completed)")
            time.sleep(30)  # Wait 30 seconds before next check
        else:
            break
    
    print("\n🎉 All training processes completed!")
    print("=" * 80)
    
    # Final summary
    print("\n📋 FINAL SUMMARY")
    print("-" * 30)
    
    for model_name, config in models.items():
        print(f"\n{model_name.upper()}:")
        artifacts = check_output_dir(config['output_dir'])
        if artifacts['json_files']:
            print(f"  📊 Metrics files: {', '.join(artifacts['json_files'])}")
        if artifacts['plots']:
            print(f"  📈 Plot files: {', '.join(artifacts['plots'])}")
        print(f"  📁 Output directory: {config['output_dir']}")
        print(f"  🔬 MLflow run: {config['mlflow_run']}")

if __name__ == "__main__":
    main()
