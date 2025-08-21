#!/usr/bin/env python3
"""
Advanced Training Monitor with Notifications
Comprehensive monitoring system for SVM, BiLSTM, and BERT training with:
- Real-time progress tracking
- Automatic completion notifications
- Periodic status updates
- MLflow metrics monitoring
- Performance analytics
"""
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict


class TrainingMonitor:
    def __init__(self, timestamp="20250819_231757"):
        self.timestamp = timestamp
        self.models = {
            'svm': {
                'pid_file': f'logs/svm_{timestamp}.pid',
                'log_file': f'logs/svm_{timestamp}.log',
                'output_dir': f'results/model_outputs/svm_run_{timestamp}',
                'mlflow_run': 'ea2cfae5fd8f472a9283a73f8e09a530',
                'expected_duration_mins': 45,
                'status': 'running'
            },
            'bilstm': {
                'pid_file': f'logs/bilstm_{timestamp}.pid',
                'log_file': f'logs/bilstm_{timestamp}.log',
                'output_dir': f'results/model_outputs/bilstm_run_{timestamp}',
                'mlflow_run': 'd71bebb05d624f1f908e2dd54fed6971',
                'expected_duration_mins': 120,
                'status': 'running'
            },
            'bert': {
                'pid_file': f'logs/bert_{timestamp}.pid',
                'log_file': f'logs/bert_{timestamp}.log',
                'output_dir': f'results/model_outputs/bert_run_{timestamp}',
                'mlflow_run': 'a64209cd7b094e5eb0b267117afbce10',
                'expected_duration_mins': 300,
                'status': 'running'
            }
        }
        self.start_time = datetime.now()
        self.last_update = {}
        self.completed_models = set()
        self.notifications = []
        
    def log_notification(self, message: str, level: str = "INFO"):
        """Log a notification with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        notification = f"[{timestamp}] {level}: {message}"
        self.notifications.append(notification)
        print(f"🔔 {notification}")
        
    def check_process_alive(self, pid: int) -> bool:
        """Check if process is still running"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def get_process_stats(self, pid: int) -> Dict:
        """Get detailed process statistics"""
        try:
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'pid,pcpu,pmem,time,vsz,rss'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    stats = lines[1].strip().split()
                    return {
                        'pid': int(stats[0]),
                        'cpu_percent': float(stats[1]),
                        'memory_percent': float(stats[2]),
                        'runtime': stats[3],
                        'virtual_memory_kb': int(stats[4]),
                        'physical_memory_kb': int(stats[5]),
                        'status': 'running'
                    }
        except Exception:
            pass
        return {'status': 'not_running'}

    def tail_log(self, logfile: str, lines: int = 10) -> str:
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
        return ""

    def parse_bert_progress(self, log_content: str) -> Dict:
        """Parse BERT training progress from logs"""
        lines = log_content.split('\n')
        
        # Find latest progress line
        progress_info = {'steps': 0, 'total_steps': 41920, 'percentage': 0.0, 'speed': '0.0s/it'}
        latest_metrics = {}
        
        for line in reversed(lines):
            # Parse progress bar: "  X%|██████     | step/total [time<remaining, speed]"
            if '|' in line and '%' in line and 'it/s' in line or 's/it' in line:
                try:
                    # Extract percentage
                    if '%|' in line:
                        pct_part = line.split('%|')[0].strip()
                        percentage = float(pct_part.split()[-1])
                        
                    # Extract steps
                    if '/' in line:
                        steps_match = re.search(r'(\d+)/(\d+)', line)
                        if steps_match:
                            current_steps = int(steps_match.group(1))
                            total_steps = int(steps_match.group(2))
                            
                        # Extract speed
                        speed_match = re.search(r'([\d.]+)(it/s|s/it)', line)
                        if speed_match:
                            speed = f"{speed_match.group(1)}{speed_match.group(2)}"
                            
                        progress_info = {
                            'steps': current_steps,
                            'total_steps': total_steps, 
                            'percentage': percentage,
                            'speed': speed
                        }
                        break
                except:
                    pass
                    
            # Parse training metrics: {'loss': 0.446, 'grad_norm': 9.452, 'learning_rate': 1.997e-05, 'epoch': 0.0}
            if line.startswith("{'loss'") and 'epoch' in line:
                try:
                    # Clean up the line to be valid JSON
                    json_str = line.replace("'", '"')
                    latest_metrics = json.loads(json_str)
                except:
                    pass
                    
        return {'progress': progress_info, 'metrics': latest_metrics}

    def parse_svm_progress(self, log_content: str) -> Dict:
        """Parse SVM training progress from logs"""
        lines = log_content.split('\n')
        
        info = {'phase': 'grid_search', 'details': ''}
        
        for line in reversed(lines):
            if 'Fitting' in line and 'folds' in line and 'candidates' in line:
                info['details'] = line.strip()
                break
            elif 'Best params' in line:
                info['phase'] = 'completed_grid_search'
                info['details'] = line.strip()
                break
            elif 'SVM val metrics' in line or 'SVM test metrics' in line:
                info['phase'] = 'evaluation'
                info['details'] = line.strip()
                break
                
        return info

    def parse_bilstm_progress(self, log_content: str) -> Dict:
        """Parse BiLSTM training progress from logs"""
        if not log_content.strip():
            return {'phase': 'starting', 'details': 'Initializing...'}
            
        lines = log_content.split('\n')
        info = {'phase': 'training', 'details': 'Processing...'}
        
        for line in reversed(lines):
            if 'epoch' in line.lower():
                info['details'] = line.strip()
                break
            elif 'loss' in line.lower():
                info['details'] = line.strip() 
                break
                
        return info

    def check_artifacts(self, output_dir: str) -> Dict:
        """Check what artifacts have been created"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return {"created": False, "files": []}
        
        json_files = list(output_path.glob("*.json"))
        png_files = list(output_path.glob("*.png"))
        npy_files = list(output_path.glob("*.npy"))
        
        return {
            "created": True,
            "json_files": [f.name for f in json_files],
            "plots": [f.name for f in png_files],
            "arrays": [f.name for f in npy_files],
            "total_files": len(list(output_path.iterdir()))
        }

    def get_model_status(self, model_name: str) -> Dict:
        """Get comprehensive status for a model"""
        config = self.models[model_name]
        
        # Check if already completed
        if model_name in self.completed_models:
            return {"status": "completed", "config": config}
            
        # Read PID
        try:
            with open(config['pid_file'], 'r') as f:
                pid = int(f.read().strip())
        except:
            return {"status": "error", "message": "Could not read PID file"}
            
        # Check if process is alive
        if not self.check_process_alive(pid):
            self.completed_models.add(model_name)
            self.log_notification(f"{model_name.upper()} training completed!", "SUCCESS")
            return {"status": "just_completed", "pid": pid, "config": config}
            
        # Get process stats
        stats = self.get_process_stats(pid)
        if stats['status'] != 'running':
            return {"status": "error", "message": "Process not found"}
            
        # Get log progress
        log_content = self.tail_log(config['log_file'], 30)
        
        # Parse progress based on model type
        if model_name == 'bert':
            progress = self.parse_bert_progress(log_content)
        elif model_name == 'svm':
            progress = self.parse_svm_progress(log_content)
        elif model_name == 'bilstm':
            progress = self.parse_bilstm_progress(log_content)
        else:
            progress = {}
            
        # Check artifacts
        artifacts = self.check_artifacts(config['output_dir'])
        
        # Calculate estimated completion
        elapsed_mins = (datetime.now() - self.start_time).total_seconds() / 60
        estimated_completion = self.start_time + timedelta(minutes=config['expected_duration_mins'])
        
        return {
            "status": "running",
            "pid": pid,
            "stats": stats,
            "progress": progress,
            "artifacts": artifacts,
            "elapsed_minutes": round(elapsed_mins, 1),
            "estimated_completion": estimated_completion.strftime("%H:%M:%S"),
            "config": config
        }

    def print_status_update(self):
        """Print comprehensive status update"""
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        
        print(f"\n{'='*80}")
        print(f"📊 TRAINING STATUS UPDATE - {current_time.strftime('%H:%M:%S')}")
        print(f"⏱️  Total Elapsed: {str(elapsed).split('.')[0]}")
        print(f"✅ Completed: {len(self.completed_models)}/3")
        print(f"{'='*80}")
        
        for model_name in ['svm', 'bilstm', 'bert']:
            print(f"\n🔸 {model_name.upper()}")
            print("-" * 40)
            
            status = self.get_model_status(model_name)
            
            if status["status"] == "completed":
                print("   ✅ COMPLETED")
                artifacts = self.check_artifacts(status["config"]["output_dir"])
                if artifacts["json_files"]:
                    print(f"   📊 Results: {', '.join(artifacts['json_files'])}")
                    
            elif status["status"] == "just_completed":
                print(f"   🏁 JUST COMPLETED (PID {status['pid']})")
                artifacts = self.check_artifacts(status["config"]["output_dir"])
                if artifacts["json_files"]:
                    print(f"   📊 Results: {', '.join(artifacts['json_files'])}")
                    
            elif status["status"] == "running":
                stats = status["stats"]
                progress = status["progress"]
                
                # Process info
                print(f"   🚀 Running (PID {status['pid']})")
                print(f"   💻 CPU: {stats['cpu_percent']}% | Memory: {stats['memory_percent']}% | Runtime: {stats['runtime']}")
                print(f"   ⏱️  Elapsed: {status['elapsed_minutes']}min | ETA: {status['estimated_completion']}")
                
                # Model-specific progress
                if model_name == 'bert' and 'progress' in progress:
                    p = progress['progress']
                    if p['steps'] > 0:
                        print(f"   📈 Progress: {p['percentage']:.1f}% ({p['steps']:,}/{p['total_steps']:,}) | Speed: {p['speed']}")
                        
                    if 'metrics' in progress and progress['metrics']:
                        m = progress['metrics']
                        print(f"   📊 Loss: {m.get('loss', 'N/A')} | LR: {m.get('learning_rate', 'N/A'):.2e} | Epoch: {m.get('epoch', 'N/A')}")
                        
                elif model_name == 'svm':
                    print(f"   📊 {progress.get('phase', 'Processing')}: {progress.get('details', 'In progress')}")
                    
                elif model_name == 'bilstm':
                    print(f"   📊 {progress.get('phase', 'Training')}: {progress.get('details', 'In progress')}")
                
                # Artifacts
                artifacts = status['artifacts']
                if artifacts['created'] and artifacts['total_files'] > 0:
                    print(f"   📁 Files created: {artifacts['total_files']}")
                    
            else:
                print(f"   ❌ Error: {status.get('message', 'Unknown error')}")

    def monitor_continuous(self, update_interval_mins: int = 2):
        """Run continuous monitoring with periodic updates"""
        self.log_notification("🔍 Advanced training monitor started", "INFO")
        self.log_notification(f"📅 Monitoring session: {self.timestamp}", "INFO")
        self.log_notification(f"⏰ Update interval: {update_interval_mins} minutes", "INFO")
        
        update_interval_seconds = update_interval_mins * 60
        next_update = time.time() + update_interval_seconds
        
        while len(self.completed_models) < 3:
            current_time = time.time()
            
            # Check for completions
            for model_name in list(self.models.keys()):
                if model_name not in self.completed_models:
                    status = self.get_model_status(model_name)
                    if status["status"] in ["completed", "just_completed"]:
                        # Model just completed - show results
                        artifacts = self.check_artifacts(status["config"]["output_dir"])
                        if artifacts.get("json_files"):
                            self.show_completion_results(model_name, artifacts)
            
            # Periodic status update
            if current_time >= next_update:
                self.print_status_update()
                next_update = current_time + update_interval_seconds
                
            time.sleep(30)  # Check every 30 seconds for completions
            
        # All completed
        self.log_notification("🎉 All training processes completed!", "SUCCESS")
        self.print_final_summary()

    def show_completion_results(self, model_name: str, artifacts: Dict):
        """Show results when a model completes"""
        print(f"\n{'🎊' * 20}")
        print(f"🏆 {model_name.upper()} TRAINING COMPLETED!")
        print(f"{'🎊' * 20}")
        
        config = self.models[model_name]
        
        # Try to read and display metrics
        for json_file in artifacts.get("json_files", []):
            if "metrics" in json_file:
                try:
                    metrics_path = Path(config["output_dir"]) / json_file
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    
                    print(f"\n📊 {json_file}:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   {key}: {value:.4f}")
                        else:
                            print(f"   {key}: {value}")
                            
                except Exception as e:
                    print(f"   ❌ Could not read {json_file}: {e}")
                    
        if artifacts.get("plots"):
            print(f"\n📈 Plots created: {', '.join(artifacts['plots'])}")
            
        print(f"\n📁 Full results in: {config['output_dir']}")
        print(f"🔬 MLflow run: {config['mlflow_run']}")

    def print_final_summary(self):
        """Print final comprehensive summary"""
        total_time = datetime.now() - self.start_time
        
        print(f"\n{'🎉' * 50}")
        print("🏁 TRAINING SESSION COMPLETE")
        print(f"{'🎉' * 50}")
        print(f"⏱️  Total Duration: {str(total_time).split('.')[0]}")
        print(f"📅 Session: {self.timestamp}")
        
        print("\n📋 FINAL RESULTS SUMMARY:")
        print("=" * 80)
        
        for model_name, config in self.models.items():
            print(f"\n🔸 {model_name.upper()}")
            print("-" * 40)
            
            artifacts = self.check_artifacts(config["output_dir"])
            
            if artifacts.get("json_files"):
                print("📊 Metrics Files:")
                for json_file in artifacts["json_files"]:
                    print(f"   • {json_file}")
                    
            if artifacts.get("plots"):
                print("📈 Visualization Files:")
                for plot_file in artifacts["plots"]:
                    print(f"   • {plot_file}")
                    
            print(f"📁 Output Directory: {config['output_dir']}")
            print(f"🔬 MLflow Run ID: {config['mlflow_run']}")
            
        print(f"\n💾 All notifications saved: {len(self.notifications)} events logged")
        print("🎯 Training session completed successfully!")

def main():
    monitor = TrainingMonitor()
    
    try:
        # Run continuous monitoring with updates every 2 minutes
        monitor.monitor_continuous(update_interval_mins=2)
        
    except KeyboardInterrupt:
        monitor.log_notification("🛑 Monitoring interrupted by user", "WARNING")
        print("\n⏸️  Monitoring stopped. Training processes are still running.")
        print("💡 Use individual monitoring commands to check progress:")
        print("   tail -f logs/*_20250819_231757.log")
        
    except Exception as e:
        monitor.log_notification(f"❌ Monitoring error: {e}", "ERROR")
        print(f"\n💥 Monitoring failed: {e}")
        print("🔧 Training processes should still be running. Check manually.")

if __name__ == "__main__":
    main()
