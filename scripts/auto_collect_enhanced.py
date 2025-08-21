#!/usr/bin/env python3
"""
Enhanced Auto Collector: Monitors all training processes and collects results automatically.
Supports MPS, CPU, and CUDA training monitoring.
"""

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_collect_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResultsCollector:
    def __init__(self):
        self.output_dirs = [
            "results/model_outputs/bert",
            "results/model_outputs/bilstm",
            "results/model_outputs/svm",
            "results/model_outputs/bert_kaggle",
            "results/model_outputs/bilstm_kaggle",
            "results/model_outputs/svm_kaggle",
            "results/model_outputs/bert_mps",
            "results/model_outputs/bilstm_mps",
            "results/model_outputs/svm_mps",
        ]
        
        self.collected = set()
        self.results_file = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.artifacts_dir = "artifacts"
        Path(self.artifacts_dir).mkdir(exist_ok=True)
        
    def check_completion(self, output_dir: str) -> bool:
        """Check if training is complete for a given output directory."""
        complete_flag = Path(output_dir) / "training_complete.flag"
        return complete_flag.exists()
    
    def collect_metrics(self, output_dir: str) -> Optional[Dict]:
        """Collect metrics from a completed training run."""
        metrics_file = Path(output_dir) / "metrics.json"
        
        if not metrics_file.exists():
            logger.warning(f"Metrics file not found: {metrics_file}")
            return None
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Add metadata
            model_name = Path(output_dir).name
            metrics['model_name'] = model_name
            metrics['output_dir'] = output_dir
            metrics['collection_time'] = datetime.now().isoformat()
            
            # Check for device info
            device_file = Path(output_dir) / "device_info.json"
            if device_file.exists():
                with open(device_file, 'r') as f:
                    device_info = json.load(f)
                    metrics['device'] = device_info.get('device', 'unknown')
                    metrics['device_name'] = device_info.get('device_name', 'unknown')
            else:
                # Infer from directory name
                if 'mps' in model_name.lower():
                    metrics['device'] = 'mps'
                    metrics['device_name'] = 'Metal Performance Shaders'
                elif 'cuda' in model_name.lower():
                    metrics['device'] = 'cuda'
                    metrics['device_name'] = 'NVIDIA CUDA'
                else:
                    metrics['device'] = 'cpu'
                    metrics['device_name'] = 'CPU'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics from {output_dir}: {e}")
            return None
    
    def save_checkpoint(self, output_dir: str):
        """Copy the best checkpoint to artifacts directory."""
        try:
            checkpoint_dir = Path(output_dir) / "checkpoint"
            if checkpoint_dir.exists():
                dest_dir = Path(self.artifacts_dir) / Path(output_dir).name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(checkpoint_dir, dest_dir)
                logger.info(f"Checkpoint saved to {dest_dir}")
            else:
                # Look for model files
                model_files = list(Path(output_dir).glob("*.pt")) + \
                              list(Path(output_dir).glob("*.pth")) + \
                              list(Path(output_dir).glob("*.pkl"))
                
                if model_files:
                    dest_dir = Path(self.artifacts_dir) / Path(output_dir).name
                    dest_dir.mkdir(exist_ok=True)
                    for model_file in model_files:
                        shutil.copy2(model_file, dest_dir)
                    logger.info(f"Model files saved to {dest_dir}")
                    
        except Exception as e:
            logger.error(f"Error saving checkpoint from {output_dir}: {e}")
    
    def generate_report(self, all_results: List[Dict]):
        """Generate a comprehensive training report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"results/training_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MPS TRAINING RESULTS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Group results by model type
            models = {}
            for result in all_results:
                model_type = result['model_name'].split('_')[0]
                if model_type not in models:
                    models[model_type] = []
                models[model_type].append(result)
            
            # Print comparison for each model type
            for model_type, results in models.items():
                f.write(f"\n{model_type.upper()} RESULTS\n")
                f.write("-" * 40 + "\n")
                
                for result in sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True):
                    f.write(f"\nConfiguration: {result['model_name']}\n")
                    f.write(f"  Device: {result.get('device', 'unknown')} ({result.get('device_name', 'unknown')})\n")
                    f.write(f"  Accuracy: {result.get('accuracy', 0):.4f}\n")
                    f.write(f"  Precision: {result.get('precision', 0):.4f}\n")
                    f.write(f"  Recall: {result.get('recall', 0):.4f}\n")
                    f.write(f"  F1-Score: {result.get('f1_score', 0):.4f}\n")
                    
                    if 'training_time' in result:
                        f.write(f"  Training Time: {result['training_time']:.2f} seconds\n")
                    if 'epochs' in result:
                        f.write(f"  Epochs: {result['epochs']}\n")
                    if 'batch_size' in result:
                        f.write(f"  Batch Size: {result['batch_size']}\n")
                
                # Calculate speedup if both CPU and MPS results exist
                cpu_result = next((r for r in results if 'cpu' in r.get('device', '').lower()), None)
                mps_result = next((r for r in results if 'mps' in r.get('device', '').lower()), None)
                
                if cpu_result and mps_result and 'training_time' in cpu_result and 'training_time' in mps_result:
                    speedup = cpu_result['training_time'] / mps_result['training_time']
                    f.write(f"\n  🚀 MPS Speedup: {speedup:.2f}x faster than CPU\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n")
            
            # Overall best model
            best_model = max(all_results, key=lambda x: x.get('accuracy', 0))
            f.write(f"\nBest Overall Model: {best_model['model_name']}\n")
            f.write(f"  Accuracy: {best_model.get('accuracy', 0):.4f}\n")
            f.write(f"  Device: {best_model.get('device', 'unknown')}\n")
            
            # Device comparison
            devices = {}
            for result in all_results:
                device = result.get('device', 'unknown')
                if device not in devices:
                    devices[device] = []
                devices[device].append(result.get('accuracy', 0))
            
            f.write("\nAverage Accuracy by Device:\n")
            for device, accuracies in devices.items():
                avg_acc = sum(accuracies) / len(accuracies)
                f.write(f"  {device}: {avg_acc:.4f} (n={len(accuracies)})\n")
        
        logger.info(f"Report saved to {report_file}")
        return report_file
    
    def trigger_visualization(self):
        """Trigger the generation of comparison charts."""
        try:
            logger.info("Generating visualization charts...")
            result = subprocess.run(
                ["python", "scripts/generate_charts.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                logger.info("Charts generated successfully")
            else:
                logger.warning(f"Chart generation failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Chart generation timed out")
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
    
    def monitor_and_collect(self):
        """Main monitoring loop."""
        logger.info("Starting enhanced auto-collection monitor...")
        logger.info(f"Monitoring directories: {self.output_dirs}")
        
        all_results = []
        check_interval = 30  # seconds
        
        while True:
            try:
                newly_completed = []
                
                for output_dir in self.output_dirs:
                    if not Path(output_dir).exists():
                        continue
                    
                    if output_dir not in self.collected and self.check_completion(output_dir):
                        logger.info(f"✅ Training complete: {output_dir}")
                        
                        # Collect metrics
                        metrics = self.collect_metrics(output_dir)
                        if metrics:
                            all_results.append(metrics)
                            newly_completed.append(output_dir)
                            
                            # Save checkpoint
                            self.save_checkpoint(output_dir)
                        
                        self.collected.add(output_dir)
                
                # If we have new results, update the comprehensive results file
                if newly_completed:
                    logger.info(f"Collected {len(newly_completed)} new results")
                    
                    # Save all results
                    with open(self.results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    logger.info(f"Results saved to {self.results_file}")
                    
                    # Generate report
                    report_file = self.generate_report(all_results)
                    
                    # Trigger visualization
                    self.trigger_visualization()
                    
                    # Run manuscript generation if all expected models are complete
                    expected_models = {'bert_mps', 'bilstm_mps', 'svm_mps'}
                    completed_models = {Path(d).name for d in self.collected}
                    
                    if expected_models.issubset(completed_models):
                        logger.info("All MPS models complete! Generating manuscript...")
                        try:
                            result = subprocess.run(
                                ["python", "scripts/generate_manuscript.py"],
                                capture_output=True,
                                text=True,
                                timeout=120
                            )
                            if result.returncode == 0:
                                logger.info("✅ Manuscript generated successfully!")
                            else:
                                logger.warning(f"Manuscript generation failed: {result.stderr}")
                        except Exception as e:
                            logger.error(f"Error generating manuscript: {e}")
                
                # Check if all directories have been processed
                if len(self.collected) == len([d for d in self.output_dirs if Path(d).exists()]):
                    logger.info("🎉 All training runs complete and collected!")
                    logger.info(f"Total models collected: {len(all_results)}")
                    logger.info(f"Results file: {self.results_file}")
                    
                    # Final summary
                    if all_results:
                        best = max(all_results, key=lambda x: x.get('accuracy', 0))
                        logger.info(f"Best model: {best['model_name']} with accuracy {best.get('accuracy', 0):.4f}")
                    
                    break
                
                # Sleep before next check
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Auto-collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)

def main():
    collector = ResultsCollector()
    collector.monitor_and_collect()

if __name__ == "__main__":
    main()
