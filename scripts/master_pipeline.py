#!/usr/bin/env python3
"""
Master pipeline orchestrator for comprehensive suicide detection experiments.
Coordinates hyperparameter optimization, training, cross-dataset evaluation,
fairness analysis, and clinical evaluation.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Install with: pip install rich")

# Setup paths
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


# Configure logging
def setup_logging(log_file: Optional[Path] = None):
    """Setup logging with optional file output and rich formatting."""
    handlers = []

    if RICH_AVAILABLE:
        handlers.append(RichHandler(rich_tracebacks=True))
    else:
        handlers.append(logging.StreamHandler())

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logging.getLogger(__name__)


class PipelineManifest:
    """Manages pipeline execution manifest for tracking artifacts."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.manifest_path = output_dir / "pipeline_manifest.json"
        self.manifest = {
            "start_time": datetime.now().isoformat(),
            "stages": {},
            "config": {},
            "artifacts": {},
        }

    def update_stage(self, stage: str, status: str, artifacts: Dict[str, str] = None):
        """Update manifest with stage execution results."""
        self.manifest["stages"][stage] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "artifacts": artifacts or {},
        }
        self.save()

    def add_artifact(self, key: str, path: str):
        """Add an artifact path to the manifest."""
        self.manifest["artifacts"][key] = path
        self.save()

    def save(self):
        """Save manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)


class MasterPipeline:
    """Master pipeline orchestrator for comprehensive experiments."""

    def __init__(self, config_path: str, output_dir: str):
        """Initialize pipeline with configuration.

        Args:
            config_path: Path to pipeline configuration YAML
            output_dir: Base output directory for all results
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = (
            self.output_dir / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.logger = setup_logging(log_file)

        # Initialize manifest
        self.manifest = PipelineManifest(self.output_dir)
        self.manifest.manifest["config"] = self.config

        # Console for rich output
        self.console = Console() if RICH_AVAILABLE else None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration from YAML."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _run_command(self, cmd: List[str], stage_name: str) -> bool:
        """Run a command and track its execution.

        Args:
            cmd: Command to execute
            stage_name: Name of the pipeline stage

        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Running {stage_name}: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.debug(f"Output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            return False

    def run_hyperopt(self) -> Dict[str, str]:
        """Run hyperparameter optimization for all models.

        Returns:
            Dictionary mapping model names to best parameter files
        """
        if not self.config.get("hyperopt", {}).get("enabled", False):
            self.logger.info("Hyperparameter optimization disabled in config")
            return {}

        self.logger.info("Starting hyperparameter optimization stage")
        artifacts = {}

        models = self.config["hyperopt"].get("models", ["svm", "bilstm", "bert"])
        n_trials = self.config["hyperopt"].get("n_trials", 50)
        timeout = self.config["hyperopt"].get("timeout", 3600)

        for model in models:
            self.logger.info(f"Optimizing {model} hyperparameters")

            cmd = [
                "python",
                "-m",
                "suicide_detection.hyperopt.optuna_runner",
                "--model",
                model,
                "--config",
                "configs/hyperopt.yaml",
                "--n_trials",
                str(n_trials),
                "--timeout",
                str(timeout),
            ]

            if self._run_command(cmd, f"hyperopt_{model}"):
                param_file = self.output_dir / "hyperopt" / f"{model}_best_params.json"
                if param_file.exists():
                    artifacts[f"{model}_params"] = str(param_file)
                    self.logger.info(f"Best parameters for {model} saved to {param_file}")

        self.manifest.update_stage("hyperopt", "completed", artifacts)
        return artifacts

    def run_baseline_training(self, hyperopt_params: Dict[str, str] = None) -> Dict[str, str]:
        """Run baseline training with best hyperparameters.

        Args:
            hyperopt_params: Dictionary mapping models to parameter files

        Returns:
            Dictionary of training artifacts
        """
        self.logger.info("Starting baseline training stage")
        artifacts = {}

        models = self.config["training"].get("models", ["svm", "bilstm", "bert"])
        datasets = self.config["training"].get("datasets", ["kaggle"])

        for dataset in datasets:
            for model in models:
                self.logger.info(f"Training {model} on {dataset}")

                output_dir = self.output_dir / "model_outputs" / f"{model}_{dataset}"

                cmd = [
                    "python",
                    "-m",
                    "suicide_detection.training.train",
                    "--model",
                    model,
                    "--dataset",
                    dataset,
                    "--output_dir",
                    str(output_dir),
                ]

                # Add hyperparameter file if available
                if hyperopt_params and f"{model}_params" in hyperopt_params:
                    cmd.extend(["--param_file", hyperopt_params[f"{model}_params"]])

                if self._run_command(cmd, f"train_{model}_{dataset}"):
                    artifacts[f"{model}_{dataset}_model"] = str(output_dir)

        self.manifest.update_stage("baseline_training", "completed", artifacts)
        return artifacts

    def run_cross_dataset_evaluation(self) -> Dict[str, str]:
        """Run cross-dataset evaluation.

        Returns:
            Dictionary of cross-dataset artifacts
        """
        self.logger.info("Starting cross-dataset evaluation stage")

        train_datasets = self.config["cross_dataset"].get("train_datasets", ["kaggle"])
        test_datasets = self.config["cross_dataset"].get("test_datasets", ["kaggle", "mendeley"])
        models = self.config["cross_dataset"].get("models", ["svm", "bilstm", "bert"])

        output_dir = self.output_dir / "cross_dataset"

        cmd = (
            ["python", "scripts/cross_dataset_evaluate.py", "--train_datasets"]
            + train_datasets
            + ["--test_datasets"]
            + test_datasets
            + ["--models"]
            + models
            + ["--output_dir", str(output_dir), "--use_cached"]
        )

        artifacts = {}
        if self._run_command(cmd, "cross_dataset_eval"):
            artifacts["results"] = str(output_dir / "cross_dataset_results.json")
            artifacts["analysis"] = str(output_dir / "cross_dataset_analysis.md")

            # Add heatmaps
            for model in models:
                heatmap = output_dir / f"{model}_cross_dataset_heatmap.png"
                if heatmap.exists():
                    artifacts[f"{model}_heatmap"] = str(heatmap)

        self.manifest.update_stage("cross_dataset", "completed", artifacts)
        return artifacts

    def run_transformer_variants(self) -> Dict[str, str]:
        """Run experiments with different transformer variants.

        Returns:
            Dictionary of transformer variant artifacts
        """
        self.logger.info("Starting transformer variants stage")
        artifacts = {}

        variants = self.config.get("transformers", {}).get(
            "variants",
            [
                "bert-base-uncased",
                "roberta-base",
                "distilbert-base-uncased",
                "mental/mental-bert-base-uncased",
            ],
        )

        dataset = self.config.get("transformers", {}).get("dataset", "kaggle")

        for variant in variants:
            self.logger.info(f"Training {variant}")

            safe_name = variant.replace("/", "_")
            output_dir = self.output_dir / "transformer_variants" / safe_name

            cmd = [
                "python",
                "-m",
                "suicide_detection.training.train",
                "--model",
                "bert",
                "--dataset",
                dataset,
                "--bert_model_name",
                variant,
                "--output_dir",
                str(output_dir),
            ]

            if self._run_command(cmd, f"transformer_{safe_name}"):
                artifacts[safe_name] = str(output_dir)

        # Generate comparison table
        self._generate_transformer_comparison(artifacts)

        self.manifest.update_stage("transformer_variants", "completed", artifacts)
        return artifacts

    def _generate_transformer_comparison(self, variant_dirs: Dict[str, str]):
        """Generate comparison table for transformer variants."""
        comparison = []

        for variant, dir_path in variant_dirs.items():
            metrics_file = Path(dir_path) / "bert_test_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    comparison.append(
                        {
                            "variant": variant,
                            "accuracy": metrics.get("accuracy", 0),
                            "f1": metrics.get("f1", 0),
                            "roc_auc": metrics.get("roc_auc", 0),
                        }
                    )

        # Save comparison
        comparison_file = self.output_dir / "transformer_variants" / "comparison.json"
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)

        # Display table if rich is available
        if self.console and comparison:
            table = Table(title="Transformer Variant Comparison")
            table.add_column("Variant", style="cyan")
            table.add_column("Accuracy", justify="right")
            table.add_column("F1", justify="right")
            table.add_column("ROC-AUC", justify="right")

            for item in comparison:
                table.add_row(
                    item["variant"],
                    f"{item['accuracy']:.3f}",
                    f"{item['f1']:.3f}",
                    f"{item['roc_auc']:.3f}",
                )

            self.console.print(table)

    def run_fairness_analysis(self) -> Dict[str, str]:
        """Run fairness analysis on trained models.

        Returns:
            Dictionary of fairness analysis artifacts
        """
        self.logger.info("Starting fairness analysis stage")
        artifacts = {}

        # This would integrate with the fairness module
        # For now, placeholder implementation
        self.logger.info("Fairness analysis implementation pending full integration")

        self.manifest.update_stage("fairness", "completed", artifacts)
        return artifacts

    def run_clinical_evaluation(self) -> Dict[str, str]:
        """Run clinical-grade evaluation.

        Returns:
            Dictionary of clinical evaluation artifacts
        """
        self.logger.info("Starting clinical evaluation stage")
        artifacts = {}

        # This would integrate with enhanced metrics
        # For now, placeholder implementation
        self.logger.info("Clinical evaluation implementation pending full integration")

        self.manifest.update_stage("clinical_eval", "completed", artifacts)
        return artifacts

    def generate_final_report(self) -> str:
        """Generate comprehensive HTML report.

        Returns:
            Path to final report
        """
        self.logger.info("Generating final report")

        # This would use Jinja2 templates
        # For now, create a simple summary
        report_path = self.output_dir / "final_report.html"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Suicide Detection Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .section {{ margin: 20px 0; }}
                .artifact {{ background: #f0f0f0; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Suicide Detection Pipeline Report</h1>
            <p>Generated: {datetime.now().isoformat()}</p>
            
            <div class="section">
                <h2>Pipeline Stages</h2>
                {self._format_stages_html()}
            </div>
            
            <div class="section">
                <h2>Artifacts</h2>
                {self._format_artifacts_html()}
            </div>
        </body>
        </html>
        """

        with open(report_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Final report saved to {report_path}")
        return str(report_path)

    def _format_stages_html(self) -> str:
        """Format stages for HTML report."""
        html = "<ul>"
        for stage, info in self.manifest.manifest.get("stages", {}).items():
            status = info.get("status", "unknown")
            html += f"<li>{stage}: {status}</li>"
        html += "</ul>"
        return html

    def _format_artifacts_html(self) -> str:
        """Format artifacts for HTML report."""
        html = "<div>"
        for key, path in self.manifest.manifest.get("artifacts", {}).items():
            html += f'<div class="artifact">{key}: {path}</div>'
        html += "</div>"
        return html

    def run(self):
        """Execute the complete pipeline."""
        self.logger.info("Starting master pipeline execution")
        start_time = time.time()

        stages = self.config.get("stages", {})

        try:
            # 1. Hyperparameter optimization
            if stages.get("hyperopt", True):
                hyperopt_params = self.run_hyperopt()
            else:
                hyperopt_params = {}

            # 2. Baseline training
            if stages.get("baseline_training", True):
                training_artifacts = self.run_baseline_training(hyperopt_params)

            # 3. Cross-dataset evaluation
            if stages.get("cross_dataset", True):
                cross_dataset_artifacts = self.run_cross_dataset_evaluation()

            # 4. Transformer variants
            if stages.get("transformer_variants", True):
                transformer_artifacts = self.run_transformer_variants()

            # 5. Fairness analysis
            if stages.get("fairness", True):
                fairness_artifacts = self.run_fairness_analysis()

            # 6. Clinical evaluation
            if stages.get("clinical_eval", True):
                clinical_artifacts = self.run_clinical_evaluation()

            # 7. Generate final report
            report_path = self.generate_final_report()
            self.manifest.add_artifact("final_report", report_path)

            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")

            # Display summary
            if self.console:
                self.console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
                self.console.print(f"[yellow]Time elapsed: {elapsed_time:.2f} seconds[/yellow]")
                self.console.print(f"[cyan]Report: {report_path}[/cyan]")
                self.console.print(f"[cyan]Manifest: {self.manifest.manifest_path}[/cyan]")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.manifest.update_stage("pipeline", "failed", {"error": str(e)})
            raise


def main():
    """CLI entry point for master pipeline."""
    parser = argparse.ArgumentParser(
        description="Master pipeline orchestrator for suicide detection experiments"
    )
    parser.add_argument(
        "--config", default="configs/pipeline.yaml", help="Pipeline configuration file"
    )
    parser.add_argument(
        "--output_dir", default="results/pipeline_run", help="Output directory for pipeline results"
    )
    parser.add_argument("--stages", nargs="+", help="Specific stages to run (default: all)")

    args = parser.parse_args()

    # Create default config if it doesn't exist
    config_path = Path(args.config)
    if not config_path.exists():
        default_config = {
            "stages": {
                "hyperopt": True,
                "baseline_training": True,
                "cross_dataset": True,
                "transformer_variants": True,
                "fairness": True,
                "clinical_eval": True,
            },
            "hyperopt": {
                "enabled": True,
                "models": ["svm", "bilstm", "bert"],
                "n_trials": 50,
                "timeout": 3600,
            },
            "training": {"models": ["svm", "bilstm", "bert"], "datasets": ["kaggle"]},
            "cross_dataset": {
                "train_datasets": ["kaggle"],
                "test_datasets": ["kaggle", "mendeley"],
                "models": ["svm", "bilstm", "bert"],
            },
            "transformers": {
                "variants": [
                    "bert-base-uncased",
                    "roberta-base",
                    "distilbert-base-uncased",
                    "mental/mental-bert-base-uncased",
                ],
                "dataset": "kaggle",
            },
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default config at {config_path}")

    # Run pipeline
    pipeline = MasterPipeline(str(config_path), args.output_dir)

    # Override stages if specified
    if args.stages:
        for stage in pipeline.config["stages"]:
            pipeline.config["stages"][stage] = stage in args.stages

    pipeline.run()


if __name__ == "__main__":
    main()
