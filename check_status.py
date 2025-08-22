#!/usr/bin/env python3
"""
Quick Training Status Check
Run this anytime to get current training progress
"""
import sys

from advanced_monitor import TrainingMonitor


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        # Detailed status
        monitor = TrainingMonitor()
        monitor.print_status_update()
    else:
        # Quick status
        monitor = TrainingMonitor()
        print("🚀 Quick Training Status:")
        print("=" * 50)

        for model_name in ["svm", "bilstm", "bert"]:
            status = monitor.get_model_status(model_name)

            if status["status"] == "completed":
                print(f"✅ {model_name.upper()}: COMPLETED")
            elif status["status"] == "running":
                stats = status["stats"]
                progress = status["progress"]

                print(f"🔄 {model_name.upper()}: Running ({status['elapsed_minutes']}min)")
                print(f"   CPU: {stats['cpu_percent']}% | ETA: {status['estimated_completion']}")

                if model_name == "bert" and "progress" in progress:
                    p = progress["progress"]
                    if p["steps"] > 0:
                        print(
                            f"   Progress: {p['percentage']:.1f}% ({p['steps']:,}/{p['total_steps']:,})"
                        )
            else:
                print(f"❌ {model_name.upper()}: {status.get('message', 'Error')}")


if __name__ == "__main__":
    main()
