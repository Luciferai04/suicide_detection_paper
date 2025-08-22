#!/usr/bin/env python3
"""
Master orchestration script for MPS training pipeline.
Manages all aspects of training, monitoring, and reporting.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


class TrainingOrchestrator:
    def __init__(self, mode="full"):
        self.mode = mode
        self.processes = {}
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")

        # Directories
        self.logs_dir = Path("logs")
        self.results_dir = Path("results")
        self.run_state_dir = Path("run_state")

        # Create necessary directories
        for dir_path in [self.logs_dir, self.results_dir, self.run_state_dir]:
            dir_path.mkdir(exist_ok=True)

        # Session file
        self.session_file = self.run_state_dir / f"session_{self.session_id}.json"

    def save_session(self):
        """Save current session state."""
        session_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "mode": self.mode,
            "processes": {name: pid for name, pid in self.processes.items()},
            "status": "running",
        }

        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)

    def start_training(self, models=["bert", "bilstm", "svm"], device="mps"):
        """Start training for specified models."""
        print(f"\n🚀 Starting {device.upper()} training for models: {models}")

        for model in models:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.logs_dir / f"{model}_{device}_{timestamp}.log"

            cmd = [
                "python",
                "-m",
                "suicide_detection.training.train",
                "--model",
                model,
                "--dataset",
                "kaggle",
                "--output_dir",
                f"results/model_outputs/{model}_{device}",
                "--prefer_device",
                device,
                "--config",
                "configs/mps.yaml",
            ]

            # Start process
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.7"},
                )

            self.processes[f"{model}_{device}"] = process.pid
            print(f"  ✅ Started {model.upper()} training (PID: {process.pid})")
            print(f"     Log: {log_file}")

            # Small delay between launches
            time.sleep(2)

        self.save_session()

    def start_monitoring(self):
        """Start all monitoring processes."""
        print("\n📊 Starting monitoring services...")

        # 1. Watchdog
        if self.mode in ["full", "monitor"]:
            log_file = self.logs_dir / f"watchdog_{self.session_id}.log"
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    ["python", "scripts/watchdog.py"], stdout=log, stderr=subprocess.STDOUT
                )
            self.processes["watchdog"] = process.pid
            print(f"  ✅ Watchdog started (PID: {process.pid})")

        # 2. Auto-collector
        log_file = self.logs_dir / f"auto_collect_{self.session_id}.log"
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                ["python", "scripts/auto_collect_enhanced.py"], stdout=log, stderr=subprocess.STDOUT
            )
        self.processes["auto_collect"] = process.pid
        print(f"  ✅ Auto-collector started (PID: {process.pid})")

        # 3. Dashboard server
        if self.mode in ["full", "dashboard"]:
            log_file = self.logs_dir / f"dashboard_{self.session_id}.log"
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    ["python", "scripts/dashboard.py"], stdout=log, stderr=subprocess.STDOUT
                )
            self.processes["dashboard"] = process.pid
            print(f"  ✅ Dashboard server started (PID: {process.pid})")
            print("     View at: http://localhost:7000/dashboard.html")

        self.save_session()

    def wait_for_completion(self):
        """Wait for all training to complete."""
        print("\n⏳ Waiting for training to complete...")
        print("   Press Ctrl+C to stop monitoring (training will continue)")

        try:
            while True:
                # Check training processes
                training_pids = [
                    pid
                    for name, pid in self.processes.items()
                    if name.startswith(("bert", "bilstm", "svm"))
                ]

                alive_count = 0
                for pid in training_pids:
                    try:
                        os.kill(pid, 0)
                        alive_count += 1
                    except OSError:
                        pass

                if alive_count == 0:
                    print("\n✅ All training processes completed!")
                    break

                # Status update
                elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                print(
                    f"\r   Status: {alive_count}/{len(training_pids)} models still training "
                    f"(elapsed: {elapsed:.1f} min)",
                    end="",
                    flush=True,
                )

                time.sleep(30)

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring stopped (training continues in background)")
            return False

        return True

    def generate_final_report(self):
        """Generate final reports and visualizations."""
        print("\n📈 Generating final reports...")

        # 1. Collect any remaining results
        print("  Collecting final results...")
        subprocess.run(["python", "collect_results.py"], capture_output=True)

        # 2. Generate charts
        print("  Generating comparison charts...")
        result = subprocess.run(
            ["python", "scripts/generate_charts.py"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("    ✅ Charts generated")

        # 3. Generate manuscript
        print("  Generating manuscript draft...")
        result = subprocess.run(
            ["python", "scripts/generate_manuscript.py"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("    ✅ Manuscript generated")

        # 4. Find latest results
        result_files = list(self.results_dir.glob("training_results_*.json"))
        if result_files:
            latest_results = max(result_files, key=lambda p: p.stat().st_mtime)

            with open(latest_results, "r") as f:
                results = json.load(f)

            # Print summary
            print("\n" + "=" * 80)
            print("FINAL SUMMARY")
            print("=" * 80)

            if results:
                # Best model
                best = max(results, key=lambda x: x.get("accuracy", 0))
                print(f"\n🏆 Best Model: {best['model_name']}")
                print(f"   Accuracy: {best.get('accuracy', 0):.4f}")
                print(f"   Device: {best.get('device', 'unknown')}")

                # Device comparison
                devices = {}
                for r in results:
                    device = r.get("device", "unknown")
                    if device not in devices:
                        devices[device] = []
                    devices[device].append(r.get("accuracy", 0))

                print("\n📊 Average Accuracy by Device:")
                for device, accs in devices.items():
                    avg = sum(accs) / len(accs) if accs else 0
                    print(f"   {device}: {avg:.4f} (n={len(accs)})")

            print("\n📁 Results saved to:")
            print(f"   JSON: {latest_results}")
            print(f"   Charts: {self.results_dir / 'plots'}")
            print(f"   Reports: {self.results_dir}")

    def cleanup(self):
        """Clean up processes on exit."""
        print("\n🧹 Cleaning up...")

        for name, pid in self.processes.items():
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"  Stopped {name} (PID: {pid})")
            except:
                pass

        # Update session status
        if self.session_file.exists():
            with open(self.session_file, "r") as f:
                session_data = json.load(f)
            session_data["status"] = "completed"
            session_data["end_time"] = datetime.now().isoformat()
            with open(self.session_file, "w") as f:
                json.dump(session_data, f, indent=2)

    def run(self):
        """Run the complete orchestration."""
        try:
            print("=" * 80)
            print("MPS TRAINING ORCHESTRATOR")
            print(f"Session: {self.session_id}")
            print(f"Mode: {self.mode}")
            print("=" * 80)

            if self.mode in ["full", "train"]:
                # Start training
                self.start_training()

            if self.mode in ["full", "monitor", "dashboard"]:
                # Start monitoring
                self.start_monitoring()

            if self.mode in ["full", "train"]:
                # Wait for completion
                completed = self.wait_for_completion()

                if completed:
                    # Generate reports
                    self.generate_final_report()

            print("\n✨ Orchestration complete!")

        except KeyboardInterrupt:
            print("\n\n⚠️  Orchestration interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            if self.mode != "dashboard":
                self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="MPS Training Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["full", "train", "monitor", "dashboard"],
        default="full",
        help="Orchestration mode",
    )
    parser.add_argument(
        "--models", nargs="+", default=["bert", "bilstm", "svm"], help="Models to train"
    )
    parser.add_argument(
        "--device", default="mps", choices=["mps", "cuda", "cpu"], help="Preferred device"
    )

    args = parser.parse_args()

    orchestrator = TrainingOrchestrator(mode=args.mode)

    if args.mode in ["full", "train"]:
        orchestrator.start_training(models=args.models, device=args.device)
        orchestrator.start_monitoring()
        orchestrator.wait_for_completion()
        orchestrator.generate_final_report()
    elif args.mode == "monitor":
        orchestrator.start_monitoring()
        orchestrator.wait_for_completion()
    elif args.mode == "dashboard":
        orchestrator.start_monitoring()
        print("\nDashboard running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    orchestrator.cleanup()


if __name__ == "__main__":
    main()
