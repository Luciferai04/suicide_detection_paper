#!/usr/bin/env python3
"""
Real-time training dashboard with web interface.
Monitors all training processes and displays metrics.
"""

import os
import re
import socketserver
import subprocess
import threading
import time
from datetime import datetime

# For web server
from http.server import SimpleHTTPRequestHandler
from typing import Dict, Optional


class TrainingMonitor:
    def __init__(self):
        self.metrics = {
            "bert": {"loss": [], "accuracy": [], "speed": [], "progress": 0},
            "bilstm": {"loss": [], "accuracy": [], "speed": [], "progress": 0},
            "svm": {"accuracy": [], "progress": 0}
        }
        
        self.log_files = {
            "bert": "logs/bert_mps_retry_20250820_021810.log",
            "bilstm": "logs/bilstm_mps_retry_20250820_021813.log",
            "svm": "logs/svm_mps_retry_20250820_021818.log"
        }
        
        self.status = {
            "bert": "initializing",
            "bilstm": "initializing", 
            "svm": "initializing"
        }
        
        self.start_times = {}
        self.completion_flags = {}
        
        # Max points to keep in memory
        self.max_points = 100
        
    def parse_bert_log(self, line: str) -> Optional[Dict]:
        """Parse BERT training log line."""
        # Match progress: "34%|███▎      | 1245/3699 [05:55<11:20,  3.60it/s]"
        progress_match = re.search(r'(\d+)%\|.*\|\s*(\d+)/(\d+)\s*\[.*,\s*([\d.]+)it/s\]', line)
        if progress_match:
            return {
                "progress": int(progress_match.group(1)),
                "current_step": int(progress_match.group(2)),
                "total_steps": int(progress_match.group(3)),
                "speed": float(progress_match.group(4))
            }
        
        # Match loss
        loss_match = re.search(r'loss:\s*([\d.]+)', line, re.IGNORECASE)
        if loss_match:
            return {"loss": float(loss_match.group(1))}
        
        # Match accuracy
        acc_match = re.search(r'accuracy:\s*([\d.]+)', line, re.IGNORECASE)
        if acc_match:
            return {"accuracy": float(acc_match.group(1))}
        
        return None
    
    def parse_bilstm_log(self, line: str) -> Optional[Dict]:
        """Parse BiLSTM training log line."""
        # Similar patterns as BERT
        return self.parse_bert_log(line)
    
    def parse_svm_log(self, line: str) -> Optional[Dict]:
        """Parse SVM training log line."""
        # Match CV progress
        cv_match = re.search(r'Fitting\s+(\d+)\s+folds.*totalling\s+(\d+)\s+fits', line)
        if cv_match:
            return {"total_fits": int(cv_match.group(2))}
        
        # Match accuracy
        acc_match = re.search(r'accuracy:\s*([\d.]+)', line, re.IGNORECASE)
        if acc_match:
            return {"accuracy": float(acc_match.group(1))}
        
        return None
    
    def update_metrics(self):
        """Update metrics from log files."""
        for model, log_file in self.log_files.items():
            if not os.path.exists(log_file):
                continue
            
            try:
                # Read last 50 lines
                result = subprocess.run(
                    ['tail', '-n', '50', log_file],
                    capture_output=True,
                    text=True
                )
                
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if not line:
                        continue
                    
                    # Parse based on model
                    if model == "bert":
                        parsed = self.parse_bert_log(line)
                    elif model == "bilstm":
                        parsed = self.parse_bilstm_log(line)
                    else:
                        parsed = self.parse_svm_log(line)
                    
                    if parsed:
                        # Update metrics
                        if "progress" in parsed:
                            self.metrics[model]["progress"] = parsed["progress"]
                            self.status[model] = "training"
                        
                        if "speed" in parsed:
                            speeds = self.metrics[model].get("speed", [])
                            speeds.append(parsed["speed"])
                            if len(speeds) > self.max_points:
                                speeds.pop(0)
                            self.metrics[model]["speed"] = speeds
                        
                        if "loss" in parsed:
                            losses = self.metrics[model].get("loss", [])
                            losses.append(parsed["loss"])
                            if len(losses) > self.max_points:
                                losses.pop(0)
                            self.metrics[model]["loss"] = losses
                        
                        if "accuracy" in parsed:
                            accs = self.metrics[model].get("accuracy", [])
                            accs.append(parsed["accuracy"])
                            if len(accs) > self.max_points:
                                accs.pop(0)
                            self.metrics[model]["accuracy"] = accs
                
                # Check for completion
                output_dir = f"results/model_outputs/{model}_mps"
                complete_flag = os.path.join(output_dir, "training_complete.flag")
                if os.path.exists(complete_flag):
                    self.status[model] = "completed"
                    self.metrics[model]["progress"] = 100
                    
            except Exception as e:
                print(f"Error updating metrics for {model}: {e}")
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>MPS Training Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .models {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .model-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            color: #333;
        }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .model-name {
            font-size: 1.5em;
            font-weight: bold;
        }
        .status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }
        .status.training { background: #fbbf24; color: white; }
        .status.completed { background: #10b981; color: white; }
        .status.initializing { background: #6b7280; color: white; }
        .status.error { background: #ef4444; color: white; }
        .progress-bar {
            background: #e5e7eb;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric {
            background: #f3f4f6;
            padding: 10px;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 0.85em;
            color: #6b7280;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #1f2937;
        }
        .timestamp {
            text-align: center;
            margin-top: 30px;
            opacity: 0.8;
        }
        .alert {
            background: #fef2f2;
            border: 2px solid #ef4444;
            color: #991b1b;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 MPS Training Dashboard</h1>
"""
        
        # Add model cards
        html += '<div class="models">'
        
        for model in ["bert", "bilstm", "svm"]:
            metrics = self.metrics[model]
            status = self.status[model]
            
            html += f'''
        <div class="model-card">
            <div class="model-header">
                <div class="model-name">{model.upper()}</div>
                <div class="status {status}">{status.upper()}</div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill" style="width: {metrics["progress"]}%">
                    {metrics["progress"]}%
                </div>
            </div>
            
            <div class="metrics">
'''
            
            # Add metrics based on what's available
            if metrics.get("loss"):
                latest_loss = metrics["loss"][-1] if metrics["loss"] else 0
                html += f'''
                <div class="metric">
                    <div class="metric-label">Loss</div>
                    <div class="metric-value">{latest_loss:.4f}</div>
                </div>
'''
            
            if metrics.get("accuracy"):
                latest_acc = metrics["accuracy"][-1] if metrics["accuracy"] else 0
                html += f'''
                <div class="metric">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{latest_acc:.2%}</div>
                </div>
'''
            
            if metrics.get("speed"):
                latest_speed = metrics["speed"][-1] if metrics["speed"] else 0
                avg_speed = sum(metrics["speed"]) / len(metrics["speed"]) if metrics["speed"] else 0
                html += f'''
                <div class="metric">
                    <div class="metric-label">Speed</div>
                    <div class="metric-value">{latest_speed:.2f} it/s</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Speed</div>
                    <div class="metric-value">{avg_speed:.2f} it/s</div>
                </div>
'''
            
            html += '''
            </div>
        </div>
'''
        
        html += '</div>'
        
        # Add timestamp
        html += f'''
        <div class="timestamp">
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
'''
        
        return html
    
    def save_dashboard(self):
        """Save dashboard HTML to file."""
        self.update_metrics()
        html = self.generate_html_dashboard()
        
        with open("dashboard.html", "w") as f:
            f.write(html)
    
    def run_dashboard_server(self, port=7000):
        """Run dashboard web server."""
        print(f"🌐 Starting dashboard server on http://localhost:{port}")
        
        # Update dashboard in background
        def update_loop():
            while True:
                self.save_dashboard()
                time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        # Start web server
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")  # Go to project root
        
        Handler = SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"📊 Dashboard available at http://localhost:{port}/dashboard.html")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n👋 Dashboard server stopped")

def main():
    monitor = TrainingMonitor()
    
    # Option to just generate once or run server
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        monitor.save_dashboard()
        print("✅ Dashboard saved to dashboard.html")
    else:
        monitor.run_dashboard_server()

if __name__ == "__main__":
    main()
