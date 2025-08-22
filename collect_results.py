#!/usr/bin/env python3
"""
Comprehensive Results Collection and Reporting System

Collects training results, generates comparison reports, and creates
publication-ready summaries for suicide risk detection research.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsCollector:
    """
    Comprehensive results collector for suicide risk detection models.

    Collects metrics, plots, model artifacts, and generates
    publication-ready reports and visualizations.
    """

    def __init__(self, session_timestamp: str = "20250819_231757"):
        """
        Initialize results collector.

        Args:
            session_timestamp: Training session timestamp
        """
        self.session_timestamp = session_timestamp
        self.base_dir = Path(".")
        self.results_dir = self.base_dir / "results"
        self.models = ["svm", "bilstm", "bert"]

        # Results storage
        self.collected_results = {}
        self.comparison_metrics = {}

        logger.info(f"Initialized results collector for session {session_timestamp}")

    def collect_all_results(self) -> Dict[str, Any]:
        """
        Collect all available results from the training session.

        Returns:
            Dictionary containing all collected results
        """

        results = {
            "session_metadata": {
                "timestamp": self.session_timestamp,
                "collection_time": datetime.now().isoformat(),
                "models_trained": self.models,
            },
            "models": {},
        }

        # Collect results for each model
        for model in self.models:
            logger.info(f"Collecting results for {model}...")
            model_results = self._collect_model_results(model)
            results["models"][model] = model_results

        # Generate comparison metrics
        results["comparison"] = self._generate_comparison_metrics(results["models"])

        # Collect MLflow information
        results["mlflow"] = self._collect_mlflow_results()

        self.collected_results = results
        logger.info("Results collection completed")

        return results

    def _collect_model_results(self, model: str) -> Dict[str, Any]:
        """Collect results for a specific model."""

        model_dir = self.results_dir / "model_outputs" / f"{model}_run_{self.session_timestamp}"

        model_results = {
            "model_name": model,
            "output_directory": str(model_dir),
            "training_completed": model_dir.exists(),
            "metrics": {},
            "artifacts": {},
            "plots": [],
            "status": "unknown",
        }

        if not model_dir.exists():
            model_results["status"] = "not_started"
            logger.warning(f"No results directory found for {model}")
            return model_results

        # Check for completion indicators
        metric_files = list(model_dir.glob("*_metrics.json"))
        if metric_files:
            model_results["status"] = "completed"

            # Collect metrics files
            for metric_file in metric_files:
                split_name = metric_file.stem.replace(f"{model}_", "").replace("_metrics", "")
                try:
                    with open(metric_file, "r") as f:
                        metrics = json.load(f)
                    model_results["metrics"][split_name] = metrics
                    logger.info(f"Collected {split_name} metrics for {model}")
                except Exception as e:
                    logger.error(f"Error reading {metric_file}: {e}")
        else:
            model_results["status"] = "in_progress"

        # Collect plots
        plot_files = list(model_dir.glob("*.png"))
        model_results["plots"] = [str(p) for p in plot_files]

        # Collect other artifacts
        npy_files = list(model_dir.glob("*.npy"))
        csv_files = list(model_dir.glob("*.csv"))

        model_results["artifacts"] = {
            "probability_arrays": [str(f) for f in npy_files],
            "feature_importance": [str(f) for f in csv_files if "importance" in f.name],
            "other_files": len(list(model_dir.iterdir())),
        }

        return model_results

    def _generate_comparison_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison metrics across models."""

        comparison = {
            "completion_status": {},
            "performance_comparison": {},
            "best_model": {},
            "statistical_tests": {},
        }

        # Completion status
        for model, results in model_results.items():
            comparison["completion_status"][model] = results["status"]

        # Performance comparison (if metrics available)
        completed_models = [
            model
            for model, results in model_results.items()
            if results["status"] == "completed" and results["metrics"]
        ]

        if completed_models:
            comparison["performance_comparison"] = self._compare_performance(
                {model: model_results[model]["metrics"] for model in completed_models}
            )

            # Identify best model
            comparison["best_model"] = self._identify_best_model(
                comparison["performance_comparison"]
            )

        return comparison

    def _compare_performance(self, model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across models."""

        comparison_data = []

        for model, metrics in model_metrics.items():
            for split, split_metrics in metrics.items():
                if isinstance(split_metrics, dict):
                    row = {"model": model, "split": split}

                    # Standard metrics
                    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
                        row[metric] = split_metrics.get(metric, None)

                    # Clinical metrics
                    row["cost_weighted_accuracy"] = split_metrics.get(
                        "cost_weighted_accuracy", None
                    )

                    comparison_data.append(row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)

            # Summary statistics
            summary = {}
            for split in ["val", "test"]:
                split_data = comparison_df[comparison_df["split"] == split]
                if not split_data.empty:
                    summary[split] = {}
                    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                        if metric in split_data.columns:
                            metric_data = split_data[metric].dropna()
                            if not metric_data.empty:
                                summary[split][metric] = {
                                    "mean": float(metric_data.mean()),
                                    "std": float(metric_data.std()),
                                    "min": float(metric_data.min()),
                                    "max": float(metric_data.max()),
                                    "models": dict(zip(split_data["model"], metric_data)),
                                }

            return {"summary_statistics": summary, "detailed_comparison": comparison_data}

        return {"message": "No comparable metrics available"}

    def _identify_best_model(self, performance_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Identify best performing model."""

        if "summary_statistics" not in performance_comparison:
            return {"message": "Cannot identify best model - no metrics available"}

        best_model = {}

        # For each metric and split, find the best model
        for split, split_metrics in performance_comparison["summary_statistics"].items():
            best_model[split] = {}

            for metric, metric_data in split_metrics.items():
                if "models" in metric_data:
                    models_dict = metric_data["models"]
                    if models_dict:
                        best_model_name = max(models_dict.keys(), key=lambda k: models_dict[k] or 0)
                        best_score = models_dict[best_model_name]

                        best_model[split][metric] = {"model": best_model_name, "score": best_score}

        # Overall best model (based on test F1 score)
        test_metrics = best_model.get("test", {})
        if "f1" in test_metrics:
            overall_best = {
                "model": test_metrics["f1"]["model"],
                "test_f1": test_metrics["f1"]["score"],
                "criterion": "Highest test F1 score",
            }
        elif "val" in best_model and "f1" in best_model["val"]:
            overall_best = {
                "model": best_model["val"]["f1"]["model"],
                "val_f1": best_model["val"]["f1"]["score"],
                "criterion": "Highest validation F1 score (test not available)",
            }
        else:
            overall_best = {"message": "Cannot determine overall best model"}

        best_model["overall"] = overall_best

        return best_model

    def _collect_mlflow_results(self) -> Dict[str, Any]:
        """Collect MLflow experiment information."""

        mlflow_dir = self.base_dir / "mlruns"

        if not mlflow_dir.exists():
            return {"status": "no_mlflow_data"}

        # Find the experiment directory
        experiment_dirs = [d for d in mlflow_dir.iterdir() if d.is_dir() and d.name.isdigit()]

        if not experiment_dirs:
            return {"status": "no_experiments"}

        # Use the most recent experiment
        latest_experiment = max(experiment_dirs, key=lambda d: d.stat().st_mtime)

        # Collect run information
        runs = []
        for run_dir in latest_experiment.iterdir():
            if run_dir.is_dir() and run_dir.name not in [".trash"]:
                run_info = self._collect_mlflow_run_info(run_dir)
                if run_info:
                    runs.append(run_info)

        return {
            "status": "available",
            "experiment_id": latest_experiment.name,
            "total_runs": len(runs),
            "runs": runs,
        }

    def _collect_mlflow_run_info(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """Collect information from an MLflow run."""

        try:
            # Read run metadata
            meta_file = run_dir / "meta.yaml"
            tags_dir = run_dir / "tags"
            metrics_dir = run_dir / "metrics"

            run_info = {"run_id": run_dir.name, "tags": {}, "metrics": {}}

            # Read tags
            if tags_dir.exists():
                for tag_file in tags_dir.glob("*"):
                    try:
                        with open(tag_file, "r") as f:
                            run_info["tags"][tag_file.name] = f.read().strip()
                    except Exception:
                        pass

            # Read metrics
            if metrics_dir.exists():
                for metric_file in metrics_dir.glob("*"):
                    try:
                        with open(metric_file, "r") as f:
                            lines = f.readlines()
                            if lines:
                                # Get the last (most recent) value
                                last_line = lines[-1].strip()
                                if last_line:
                                    parts = last_line.split()
                                    if len(parts) >= 2:
                                        run_info["metrics"][metric_file.name] = float(parts[1])
                    except Exception:
                        pass

            return run_info

        except Exception as e:
            logger.error(f"Error reading MLflow run {run_dir}: {e}")
            return None

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive training report."""

        if not self.collected_results:
            self.collect_all_results()

        report_lines = []

        # Header
        report_lines.extend(
            [
                "=" * 80,
                "SUICIDE RISK DETECTION - COMPREHENSIVE TRAINING REPORT",
                "=" * 80,
                f"Training Session: {self.session_timestamp}",
                f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Models Trained: {', '.join(self.models)}",
                "",
            ]
        )

        # Training Summary
        report_lines.extend(
            [
                "TRAINING SUMMARY",
                "-" * 40,
            ]
        )

        for model in self.models:
            model_data = self.collected_results["models"][model]
            status = model_data["status"]

            report_lines.append(f"{model.upper()}: {status.upper()}")

            if status == "completed" and model_data["metrics"]:
                # Show key metrics
                if "test" in model_data["metrics"]:
                    test_metrics = model_data["metrics"]["test"]
                    if isinstance(test_metrics, dict):
                        f1 = test_metrics.get("f1", "N/A")
                        auc = test_metrics.get("roc_auc", "N/A")
                        report_lines.append(
                            f"  Test F1: {f1:.4f}, AUC: {auc:.4f}"
                            if f1 != "N/A"
                            else "  Metrics available"
                        )

            elif status == "in_progress":
                artifacts = model_data["artifacts"]["other_files"]
                report_lines.append(f"  Progress: {artifacts} files created")

            elif status == "not_started":
                report_lines.append("  Status: Training not initiated")

        report_lines.append("")

        # Performance Comparison
        if "performance_comparison" in self.collected_results["comparison"]:
            comparison = self.collected_results["comparison"]["performance_comparison"]

            if "summary_statistics" in comparison:
                report_lines.extend(
                    [
                        "PERFORMANCE COMPARISON",
                        "-" * 40,
                    ]
                )

                for split in ["val", "test"]:
                    if split in comparison["summary_statistics"]:
                        split_data = comparison["summary_statistics"][split]
                        report_lines.append(f"{split.upper()} SET RESULTS:")

                        for metric in ["f1", "roc_auc", "accuracy", "precision", "recall"]:
                            if metric in split_data:
                                metric_info = split_data[metric]
                                models_scores = metric_info["models"]

                                report_lines.append(f"  {metric.upper()}:")
                                for model, score in models_scores.items():
                                    report_lines.append(f"    {model}: {score:.4f}")

                        report_lines.append("")

        # Best Model
        if "best_model" in self.collected_results["comparison"]:
            best_model = self.collected_results["comparison"]["best_model"]

            if "overall" in best_model and "model" in best_model["overall"]:
                overall_best = best_model["overall"]
                report_lines.extend(
                    [
                        "BEST PERFORMING MODEL",
                        "-" * 40,
                        f"Model: {overall_best['model'].upper()}",
                        f"Criterion: {overall_best['criterion']}",
                    ]
                )

                for key, value in overall_best.items():
                    if key not in ["model", "criterion"]:
                        report_lines.append(f"{key}: {value:.4f}")

                report_lines.append("")

        # Dataset Information
        report_lines.extend(
            [
                "DATASET INFORMATION",
                "-" * 40,
                "Training Data: Kaggle Suicide Detection Dataset",
                "Splits: 70% train, 15% validation, 15% test",
                "Preprocessing: Text cleaning, tokenization, stratified sampling",
                "Class Balance: Handled via weighted loss functions",
                "",
            ]
        )

        # Model Architectures
        report_lines.extend(
            [
                "MODEL ARCHITECTURES",
                "-" * 40,
                "SVM: TF-IDF features + Support Vector Machine with RBF kernel",
                "BiLSTM: Bidirectional LSTM with attention mechanism",
                "BERT: Fine-tuned BERT-base-uncased transformer model",
                "",
            ]
        )

        # Training Configuration
        report_lines.extend(
            [
                "TRAINING CONFIGURATION",
                "-" * 40,
                "Environment: CPU-only (MPS disabled for stability)",
                "Hardware: MacOS with M-series processor",
                "Framework: PyTorch + HuggingFace Transformers + scikit-learn",
                "Monitoring: MLflow experiment tracking enabled",
                "",
            ]
        )

        # Clinical Considerations
        report_lines.extend(
            [
                "CLINICAL CONSIDERATIONS",
                "-" * 40,
                "• Models designed for clinical decision support, not diagnosis",
                "• Ethical guidelines followed throughout development",
                "• Bias auditing framework implemented",
                "• Privacy protection measures in place",
                "• Interpretability features for clinical transparency",
                "",
            ]
        )

        # Next Steps
        report_lines.extend(
            [
                "NEXT STEPS",
                "-" * 40,
                "1. Complete all model training and collect final metrics",
                "2. Perform comprehensive model comparison and statistical testing",
                "3. Conduct clinical validation with healthcare professionals",
                "4. Implement real-world deployment pipeline",
                "5. Prepare results for publication",
                "",
            ]
        )

        # File Locations
        report_lines.extend(
            [
                "RESULT FILE LOCATIONS",
                "-" * 40,
            ]
        )

        for model in self.models:
            model_dir = f"results/model_outputs/{model}_run_{self.session_timestamp}/"
            report_lines.extend(
                [
                    f"{model.upper()}:",
                    f"  Directory: {model_dir}",
                    f"  Metrics: {model_dir}*_metrics.json",
                    f"  Plots: {model_dir}*.png",
                    "",
                ]
            )

        report_lines.extend(
            [
                "MLflow Experiments: mlruns/",
                f"Training Logs: logs/*_{self.session_timestamp}.log",
                "Monitoring Output: monitor_output.log",
                "",
            ]
        )

        # Footer
        report_lines.extend(
            [
                "=" * 80,
                "END OF REPORT",
                "Generated by Suicide Risk Detection Research Framework v1.0",
                "For questions or issues, contact the research team.",
                "=" * 80,
            ]
        )

        return "\\n".join(report_lines)

    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save comprehensive report to file."""

        if filename is None:
            filename = f"training_report_{self.session_timestamp}.txt"

        report_path = self.results_dir / filename
        report_content = self.generate_comprehensive_report()

        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"Report saved to {report_path}")
        return report_path

    def export_results_json(self, filename: Optional[str] = None) -> Path:
        """Export all collected results as JSON."""

        if not self.collected_results:
            self.collect_all_results()

        if filename is None:
            filename = f"training_results_{self.session_timestamp}.json"

        json_path = self.results_dir / filename

        with open(json_path, "w") as f:
            json.dump(self.collected_results, f, indent=2, default=str)

        logger.info(f"Results exported to {json_path}")
        return json_path

    def create_performance_visualizations(self) -> List[Path]:
        """Create performance visualization plots."""

        if not self.collected_results:
            self.collect_all_results()

        plots_created = []

        # Model comparison plots
        comparison = self.collected_results["comparison"]["performance_comparison"]

        if "detailed_comparison" in comparison:
            comparison_df = pd.DataFrame(comparison["detailed_comparison"])

            # Performance comparison bar plot
            plt.figure(figsize=(12, 8))

            metrics_to_plot = ["f1", "roc_auc", "accuracy", "precision", "recall"]
            test_data = comparison_df[comparison_df["split"] == "test"]

            if not test_data.empty:
                x_pos = np.arange(len(metrics_to_plot))
                width = 0.25

                for i, model in enumerate(test_data["model"].unique()):
                    model_data = test_data[test_data["model"] == model]

                    if not model_data.empty:
                        values = [
                            (
                                model_data[metric].iloc[0]
                                if metric in model_data.columns
                                and not pd.isna(model_data[metric].iloc[0])
                                else 0
                            )
                            for metric in metrics_to_plot
                        ]

                        plt.bar(x_pos + i * width, values, width, label=model.upper())

                plt.xlabel("Metrics")
                plt.ylabel("Score")
                plt.title("Model Performance Comparison (Test Set)")
                plt.xticks(x_pos + width, metrics_to_plot)
                plt.legend()
                plt.grid(True, alpha=0.3)

                plot_path = self.results_dir / f"model_comparison_{self.session_timestamp}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                plots_created.append(plot_path)
                logger.info(f"Created comparison plot: {plot_path}")

        return plots_created


def main():
    """Main execution function."""

    print("🔍 Starting comprehensive results collection...")
    print("=" * 60)

    # Initialize collector
    collector = ResultsCollector()

    # Collect all results
    results = collector.collect_all_results()

    # Generate and save report
    report_path = collector.save_report()
    print(f"📊 Comprehensive report saved: {report_path}")

    # Export JSON results
    json_path = collector.export_results_json()
    print(f"💾 Results exported to JSON: {json_path}")

    # Create visualizations
    plots = collector.create_performance_visualizations()
    if plots:
        print(f"📈 Created {len(plots)} visualization(s)")
        for plot in plots:
            print(f"   • {plot}")

    # Print summary
    print("\\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("-" * 30)

    completed_models = sum(
        1 for model_data in results["models"].values() if model_data["status"] == "completed"
    )

    in_progress_models = sum(
        1 for model_data in results["models"].values() if model_data["status"] == "in_progress"
    )

    print(f"Models completed: {completed_models}/3")
    print(f"Models in progress: {in_progress_models}/3")

    if "overall" in results["comparison"]["best_model"]:
        best = results["comparison"]["best_model"]["overall"]
        if "model" in best:
            print(f"Best model: {best['model'].upper()}")

    print("\\n📁 All results available in: results/")
    print(f"📋 Full report: {report_path.name}")
    print("\\n🎯 Results collection completed successfully!")


if __name__ == "__main__":
    main()
