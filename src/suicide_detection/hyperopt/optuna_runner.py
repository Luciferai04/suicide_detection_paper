#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for suicide detection models.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import pandas as pd
import yaml

# Add src to path for relative imports
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from suicide_detection.data_processing.load import load_dataset_secure
from suicide_detection.training.train import prepare_splits, run_bert, run_bilstm, run_svm

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for suicide detection models."""

    def __init__(self, config_path: str, storage_path: Optional[str] = None):
        """Initialize optimizer with configuration.

        Args:
            config_path: Path to YAML config file with hyperparameter search spaces
            storage_path: Optional path to SQLite database for study persistence
        """
        self.config = self._load_config(config_path)
        self.storage_path = storage_path or "results/hyperopt.db"
        self.results_dir = Path("results/hyperopt")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load hyperparameter configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _suggest_parameters(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for a given model type based on config."""
        params = {}
        model_config = self.config["hyperopt"][model_type]["parameters"]

        for param_name, param_config in model_config.items():
            if param_config["type"] == "log_uniform":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=True
                )
            elif param_config["type"] == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )

        return params

    def optimize(
        self,
        model_type: str,
        train_data,
        val_data,
        test_data,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization for a specific model.

        Args:
            model_type: One of 'svm', 'bilstm', 'bert'
            train_data: Training data tuple (X, y)
            val_data: Validation data tuple (X, y)
            test_data: Test data tuple (X, y)
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            study_name: Name for the Optuna study

        Returns:
            Dictionary with best parameters and optimization results
        """
        # Create study
        study_name = study_name or f"{model_type}_hyperopt"
        storage_url = f"sqlite:///{self.storage_path}"

        study = optuna.create_study(
            direction="maximize",
            storage=storage_url,
            study_name=study_name,
            load_if_exists=True,
            pruner=optuna.pruners.PatientPruner(
                optuna.pruners.MedianPruner(n_warmup_steps=5), patience=3
            ),
        )

        def objective(trial):
            """Objective function for Optuna optimization."""
            try:
                # Get hyperparameters for this trial
                params = self._suggest_parameters(trial, model_type)

                # Create temporary output directory
                trial_output = self.results_dir / f"trial_{trial.number}"
                trial_output.mkdir(parents=True, exist_ok=True)

                # Override config with trial parameters
                cfg_overrides = {model_type: params}

                # Train model with these parameters
                if model_type == "svm":
                    run_svm(
                        train_data,
                        val_data,
                        test_data,
                        trial_output,
                        logger,
                        cfg_overrides=cfg_overrides,
                    )
                elif model_type == "bilstm":
                    run_bilstm(
                        train_data,
                        val_data,
                        test_data,
                        trial_output,
                        logger,
                        cfg_overrides=cfg_overrides,
                    )
                elif model_type == "bert":
                    run_bert(
                        train_data,
                        val_data,
                        test_data,
                        trial_output,
                        logger,
                        cfg_overrides=cfg_overrides,
                    )

                # Read validation metrics
                val_metrics_file = trial_output / f"{model_type}_val_metrics.json"
                if val_metrics_file.exists():
                    with open(val_metrics_file, "r") as f:
                        metrics = json.load(f)
                    # Return F1 score as optimization target
                    return metrics.get("f1", 0.0)
                else:
                    logger.warning(f"Trial {trial.number}: No validation metrics found")
                    return 0.0

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return 0.0

        # Run optimization
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        logger.info(f"Target: {n_trials} trials, timeout: {timeout}s")

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Save results
        best_params = study.best_params
        best_value = study.best_value

        # Save best parameters
        best_params_file = self.results_dir / f"{model_type}_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=2)

        # Save study history
        study_df = study.trials_dataframe()
        study_csv = self.results_dir / f"{model_type}_study_history.csv"
        study_df.to_csv(study_csv, index=False)

        # Optimization summary
        results = {
            "model_type": model_type,
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "study_name": study_name,
            "results_files": {
                "best_params": str(best_params_file),
                "study_history": str(study_csv),
            },
        }

        logger.info(f"Optimization completed for {model_type}")
        logger.info(f"Best F1: {best_value:.4f}")
        logger.info(f"Best params: {best_params}")

        return results


def main():
    """CLI entry point for hyperparameter optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--model", required=True, choices=["svm", "bilstm", "bert"])
    parser.add_argument("--config", default="configs/hyperopt.yaml")
    parser.add_argument("--dataset", default="kaggle", help="Dataset to use")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--storage", help="SQLite database path")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load data
    dataset_path = Path(f"data/{args.dataset}/splits")
    if dataset_path.exists():
        train_df = pd.read_csv(dataset_path / "train.csv")
        val_df = pd.read_csv(dataset_path / "val.csv")
        test_df = pd.read_csv(dataset_path / "test.csv")

        train_data = (train_df["text"].values, train_df["label"].values)
        val_data = (val_df["text"].values, val_df["label"].values)
        test_data = (test_df["text"].values, test_df["label"].values)
    else:
        # Fallback to single file
        data_path = Path("data/raw/dataset.csv")
        df = load_dataset_secure(data_path)
        train_data, val_data, test_data = prepare_splits(df)

    # Initialize optimizer
    optimizer = OptunaOptimizer(config_path=args.config, storage_path=args.storage)

    # Run optimization
    results = optimizer.optimize(
        model_type=args.model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("\nOptimization completed!")
    print(f"Best F1 score: {results['best_value']:.4f}")
    print(f"Best parameters saved to: {results['results_files']['best_params']}")


if __name__ == "__main__":
    main()
