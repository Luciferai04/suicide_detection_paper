#!/usr/bin/env python3
"""
Watchdog for MPS training stability.
Monitors processes, detects crashes, and automatically recovers.
"""

import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil


class TrainingWatchdog:
    def __init__(self):
        self.config_file = "run_state/watchdog_config.json"
        self.crash_log = "run_state/crash_log.json"
        self.pid_file = "run_state/training_pids.json"
        self.max_retries = 3
        self.check_interval = 30  # seconds
        self.stall_threshold = 120  # seconds without progress
        
        # Create run_state directory
        Path("run_state").mkdir(exist_ok=True)
        
        # Load or initialize state
        self.state = self.load_state()
        self.crashes = self.load_crashes()
        
    def load_state(self) -> Dict:
        """Load watchdog state."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {
            "processes": {},
            "retry_counts": {},
            "last_progress": {}
        }
    
    def save_state(self):
        """Save watchdog state."""
        with open(self.config_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def load_crashes(self) -> List:
        """Load crash history."""
        if os.path.exists(self.crash_log):
            with open(self.crash_log, 'r') as f:
                return json.load(f)
        return []
    
    def save_crash(self, model: str, reason: str, action: str):
        """Log a crash event."""
        crash_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "reason": reason,
            "action": action
        }
        self.crashes.append(crash_entry)
        with open(self.crash_log, 'w') as f:
            json.dump(self.crashes, f, indent=2)
    
    def get_process_info(self, pid: int) -> Optional[Dict]:
        """Get process information."""
        try:
            process = psutil.Process(pid)
            return {
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "create_time": process.create_time()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def check_log_progress(self, log_file: str) -> bool:
        """Check if log file shows recent progress."""
        if not os.path.exists(log_file):
            return False
        
        # Check file modification time
        mtime = os.path.getmtime(log_file)
        age = time.time() - mtime
        
        # If file hasn't been modified in threshold time, consider stalled
        return age < self.stall_threshold
    
    def restart_training(self, model: str, config: Dict) -> Optional[int]:
        """Restart a training process with adjusted parameters."""
        retry_count = self.state["retry_counts"].get(model, 0)
        
        if retry_count >= self.max_retries:
            print(f"❌ Max retries ({self.max_retries}) reached for {model}. Falling back to CPU.")
            config["prefer_device"] = "cpu"
        
        # Reduce batch size on retry
        if retry_count > 0:
            if model == "bert":
                new_batch_size = max(4, 8 // (2 ** retry_count))
                config["batch_size"] = new_batch_size
                print(f"📉 Reducing {model} batch size to {new_batch_size}")
            elif model == "bilstm":
                new_batch_size = max(16, 64 // (2 ** retry_count))
                config["batch_size"] = new_batch_size
                print(f"📉 Reducing {model} batch size to {new_batch_size}")
        
        # Build command
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd = [
            "python", "-m", "suicide_detection.training.train",
            "--model", model,
            "--dataset", "kaggle",
            "--output_dir", f"results/model_outputs/{model}_mps",
            "--prefer_device", config.get("prefer_device", "mps"),
            "--config", "configs/mps.yaml"
        ]
        
        # Start process
        log_file = f"logs/{model}_watchdog_{timestamp}.log"
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.7"}
            )
        
        print(f"🔄 Restarted {model} training (PID: {process.pid})")
        
        # Update state
        self.state["processes"][model] = {
            "pid": process.pid,
            "log_file": log_file,
            "start_time": time.time(),
            "config": config
        }
        self.state["retry_counts"][model] = retry_count + 1
        self.save_state()
        
        return process.pid
    
    def kill_hung_mps_processes(self):
        """Kill hung Metal processes if needed."""
        try:
            # Check for hung Metal processes
            result = subprocess.run(
                ["pgrep", "-f", "MetalPerformanceShaders"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        # Check if process is using excessive resources
                        try:
                            proc = psutil.Process(int(pid))
                            if proc.cpu_percent() > 90 and proc.memory_percent() > 50:
                                print(f"⚠️  Killing hung MPS process {pid}")
                                os.kill(int(pid), signal.SIGKILL)
                                time.sleep(2)
                        except:
                            pass
        except Exception as e:
            print(f"Error checking MPS processes: {e}")
    
    def monitor_training(self):
        """Main monitoring loop."""
        print("🔍 Watchdog started. Monitoring training processes...")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Stall threshold: {self.stall_threshold}s")
        print(f"   Max retries: {self.max_retries}")
        
        # Load current PIDs
        if os.path.exists(self.pid_file):
            with open(self.pid_file, 'r') as f:
                current_pids = json.load(f)
                
                for model, pid in current_pids.items():
                    if model not in self.state["processes"]:
                        self.state["processes"][model] = {
                            "pid": pid,
                            "log_file": f"logs/{model}_mps_*.log",
                            "start_time": time.time(),
                            "config": {"prefer_device": "mps"}
                        }
        
        while True:
            try:
                for model, proc_info in list(self.state["processes"].items()):
                    pid = proc_info["pid"]
                    log_file = proc_info["log_file"]
                    
                    # Check process status
                    proc_status = self.get_process_info(pid)
                    
                    if proc_status is None:
                        # Process died
                        print(f"❌ {model} process (PID: {pid}) has died!")
                        self.save_crash(model, "Process died", "Restarting")
                        
                        # Restart with adjusted config
                        new_pid = self.restart_training(model, proc_info["config"])
                        
                    elif proc_status["status"] == "zombie":
                        # Zombie process
                        print(f"🧟 {model} process (PID: {pid}) is a zombie!")
                        os.kill(pid, signal.SIGKILL)
                        self.save_crash(model, "Zombie process", "Killed and restarting")
                        
                        # Restart
                        new_pid = self.restart_training(model, proc_info["config"])
                        
                    elif not self.check_log_progress(log_file):
                        # Check if process is stalled
                        print(f"⏸️  {model} appears to be stalled (no progress in {self.stall_threshold}s)")
                        
                        # Try gentle restart first
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(5)
                        
                        # Force kill if still alive
                        if self.get_process_info(pid):
                            os.kill(pid, signal.SIGKILL)
                        
                        self.save_crash(model, "Stalled", "Killed and restarting")
                        
                        # Restart with reduced batch size
                        new_pid = self.restart_training(model, proc_info["config"])
                    
                    else:
                        # Process is healthy
                        runtime = (time.time() - proc_info["start_time"]) / 60
                        print(f"✅ {model} is healthy (PID: {pid}, runtime: {runtime:.1f} min)")
                
                # Check for hung MPS processes periodically
                self.kill_hung_mps_processes()
                
                # Sleep before next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\n👋 Watchdog stopped by user")
                break
            except Exception as e:
                print(f"⚠️  Watchdog error: {e}")
                time.sleep(self.check_interval)

def main():
    watchdog = TrainingWatchdog()
    watchdog.monitor_training()

if __name__ == "__main__":
    main()
