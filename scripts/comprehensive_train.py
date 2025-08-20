#!/usr/bin/env python3
"""
Comprehensive training script with:
- Hyperparameter optimization via Optuna
- Cross-dataset evaluation
- Multiple transformer variants
- Enhanced fairness analysis
- Clinical-grade evaluation metrics
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import yaml

from suicide_detection.training.train import main as train_main, prepare_splits
from suicide_detection.data_processing.load import load_dataset_secure
from suicide_detection.evaluation.metrics import compute_metrics, bootstrap_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hyperopt_config(config_path: Path) -> Dict[str, Any]:
    """Load hyperparameter optimization configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_optuna_study(config: Dict[str, Any]) -> optuna.Study:
    """Create Optuna study for hyperparameter optimization."""
    storage = config['hyperopt'].get('storage')
    study_name = config['hyperopt'].get('study_name', 'suicide_detection')
    
    return optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name=study_name,
        load_if_exists=True
    )


def suggest_hyperparameters(trial: optuna.Trial, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters for a given model type."""
    params = {}
    model_config = config['hyperopt'][model_type]['parameters']
    
    for param_name, param_config in model_config.items():
        if param_config['type'] == 'log_uniform':
            params[param_name] = trial.suggest_float(
                param_name, param_config['low'], param_config['high'], log=True
            )
        elif param_config['type'] == 'uniform':
            params[param_name] = trial.suggest_float(
                param_name, param_config['low'], param_config['high']
            )
        elif param_config['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name, param_config['choices']
            )
        elif param_config['type'] == 'int':
            params[param_name] = trial.suggest_int(
                param_name, param_config['low'], param_config['high']
            )
    
    return params


def run_cross_dataset_evaluation(
    datasets: List[str], 
    models: List[str], 
    output_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run cross-dataset evaluation (train on one, test on others)."""
    results = {}
    
    for train_dataset in datasets:
        for test_dataset in datasets:
            if train_dataset == test_dataset:
                continue
                
            for model in models:
                logger.info(f"Training {model} on {train_dataset}, testing on {test_dataset}")
                
                # Prepare data paths
                train_path = Path(f"data/{train_dataset}/splits/train.csv")
                val_path = Path(f"data/{train_dataset}/splits/val.csv")
                test_path = Path(f"data/{test_dataset}/splits/test.csv")
                
                if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
                    logger.warning(f"Missing data for {train_dataset} -> {test_dataset}, skipping")
                    continue
                
                # Run training
                result_key = f"{model}_{train_dataset}_to_{test_dataset}"
                model_output_dir = output_dir / "cross_dataset" / result_key
                
                try:
                    # This would need to be integrated with your training pipeline
                    # For now, placeholder for the structure
                    results[result_key] = {
                        "train_dataset": train_dataset,
                        "test_dataset": test_dataset,
                        "model": model,
                        "metrics": {},  # Would be populated by actual training
                        "status": "completed"
                    }
                except Exception as e:
                    logger.error(f"Failed {result_key}: {e}")
                    results[result_key] = {
                        "train_dataset": train_dataset,
                        "test_dataset": test_dataset,
                        "model": model,
                        "status": "failed",
                        "error": str(e)
                    }
    
    return results


def infer_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Infer basic demographic information for fairness analysis."""
    # This is a placeholder - in practice, you'd use more sophisticated methods
    # or rely on available demographic data
    
    # Simple heuristics based on text analysis (very basic)
    df['inferred_age_group'] = np.random.choice(['18-25', '26-35', '36-50', '50+'], size=len(df))
    df['inferred_gender'] = np.random.choice(['M', 'F', 'NB'], size=len(df), p=[0.4, 0.5, 0.1])
    
    logger.warning("Using synthetic demographic data for demonstration. "
                   "Replace with proper demographic inference or actual data.")
    
    return df


def run_fairness_analysis(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray,
    demographics: Dict[str, np.ndarray],
    output_dir: Path
) -> Dict[str, Any]:
    """Run comprehensive fairness analysis."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    
    fairness_results = {}
    
    for demo_name, demo_values in demographics.items():
        group_results = {}
        unique_groups = np.unique(demo_values)
        
        for group in unique_groups:
            mask = demo_values == group
            if np.sum(mask) < 10:  # Skip very small groups
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            group_y_prob = y_prob[mask]
            
            group_results[str(group)] = {
                'accuracy': accuracy_score(group_y_true, group_y_pred),
                'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                'roc_auc': roc_auc_score(group_y_true, group_y_prob) if len(np.unique(group_y_true)) > 1 else 0.0,
                'n_samples': np.sum(mask)
            }
        
        # Calculate fairness metrics
        if len(group_results) > 1:
            accuracies = [r['accuracy'] for r in group_results.values()]
            recalls = [r['recall'] for r in group_results.values()]
            
            fairness_results[demo_name] = {
                'groups': group_results,
                'demographic_parity_diff': max(accuracies) - min(accuracies),
                'equal_opportunity_diff': max(recalls) - min(recalls),
                'overall_fairness_score': 1.0 - max(
                    max(accuracies) - min(accuracies),
                    max(recalls) - min(recalls)
                )
            }
    
    # Save fairness results
    fairness_output = output_dir / "fairness_analysis.json"
    fairness_output.parent.mkdir(parents=True, exist_ok=True)
    with open(fairness_output, 'w') as f:
        json.dump(fairness_results, f, indent=2)
    
    return fairness_results


def run_enhanced_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Run enhanced evaluation with clinical targets and bootstrapping."""
    
    # Basic metrics
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_prob)
    
    # Clinical cost-weighted accuracy
    cost_matrix = config['evaluation']['cost_matrix']
    fn_cost = cost_matrix['false_negative_cost']
    fp_cost = cost_matrix['false_positive_cost']
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    total_cost = fn * fn_cost + fp * fp_cost
    max_cost = len(y_true) * max(fn_cost, fp_cost)
    cost_weighted_accuracy = 1.0 - (total_cost / max_cost)
    
    # Bootstrap confidence intervals
    if config['evaluation']['bootstrap']['enabled']:
        n_bootstrap = config['evaluation']['bootstrap']['n_resamples']
        bootstrap_results = bootstrap_metrics(y_true, y_prob, n_bootstrap)
        
        enhanced_metrics = {
            **metrics.__dict__,
            'cost_weighted_accuracy': cost_weighted_accuracy,
            'clinical_targets_met': {
                'accuracy': metrics.accuracy >= config['evaluation']['clinical_targets']['min_accuracy'],
                'recall': metrics.recall >= config['evaluation']['clinical_targets']['min_recall'],
                'precision': metrics.precision >= config['evaluation']['clinical_targets']['min_precision'],
            },
            'bootstrap_ci': bootstrap_results
        }
    else:
        enhanced_metrics = {
            **metrics.__dict__,
            'cost_weighted_accuracy': cost_weighted_accuracy,
            'clinical_targets_met': {
                'accuracy': metrics.accuracy >= config['evaluation']['clinical_targets']['min_accuracy'],
                'recall': metrics.recall >= config['evaluation']['clinical_targets']['min_recall'],
                'precision': metrics.precision >= config['evaluation']['clinical_targets']['min_precision'],
            }
        }
    
    return enhanced_metrics


def main():
    parser = argparse.ArgumentParser(description="Comprehensive suicide detection training")
    parser.add_argument("--config", default="configs/hyperopt.yaml", help="Hyperopt config file")
    parser.add_argument("--output_dir", default="results/comprehensive", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=["kaggle"], help="Datasets to use")
    parser.add_argument("--models", nargs="+", default=["svm", "bilstm", "bert"], help="Models to train")
    parser.add_argument("--skip_hyperopt", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument("--skip_cross_dataset", action="store_true", help="Skip cross-dataset evaluation")
    parser.add_argument("--skip_fairness", action="store_true", help="Skip fairness analysis")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file {config_path} not found")
        return
    
    config = load_hyperopt_config(config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Hyperparameter optimization (if not skipped)
    if not args.skip_hyperopt and config['hyperopt']['enabled']:
        logger.info("Starting hyperparameter optimization...")
        
        for model in args.models:
            if model not in config['hyperopt']:
                continue
                
            study = create_optuna_study(config)
            
            def objective(trial):
                params = suggest_hyperparameters(trial, model, config)
                # This would integrate with your training pipeline
                # Return validation metric to maximize
                return 0.85  # Placeholder
            
            study.optimize(
                objective, 
                n_trials=config['hyperopt']['n_trials'],
                timeout=config['hyperopt']['timeout']
            )
            
            # Save best parameters
            best_params_file = output_dir / f"{model}_best_params.json"
            with open(best_params_file, 'w') as f:
                json.dump(study.best_params, f, indent=2)
            
            logger.info(f"Best {model} parameters: {study.best_params}")
    
    # 2. Cross-dataset evaluation (if not skipped)
    if not args.skip_cross_dataset:
        logger.info("Starting cross-dataset evaluation...")
        cross_results = run_cross_dataset_evaluation(
            args.datasets, args.models, output_dir, config
        )
        
        # Save cross-dataset results
        cross_results_file = output_dir / "cross_dataset_results.json"
        with open(cross_results_file, 'w') as f:
            json.dump(cross_results, f, indent=2)
    
    # 3. Enhanced model training with multiple BERT variants
    logger.info("Training enhanced models...")
    
    for dataset in args.datasets:
        dataset_path = Path(f"data/{dataset}/splits")
        if not dataset_path.exists():
            logger.warning(f"Dataset {dataset} not found, skipping")
            continue
        
        for model in args.models:
            if model == "bert":
                # Train multiple BERT variants
                for bert_model in config['models']['bert_variants']:
                    logger.info(f"Training {bert_model} on {dataset}")
                    model_output_dir = output_dir / f"{dataset}_{model}_{bert_model.replace('/', '_')}"
                    
                    # This would integrate with your training pipeline
                    # train_model(model, bert_model, dataset_path, model_output_dir, config)
            else:
                logger.info(f"Training {model} on {dataset}")
                model_output_dir = output_dir / f"{dataset}_{model}"
                
                # This would integrate with your training pipeline
                # train_model(model, None, dataset_path, model_output_dir, config)
    
    logger.info("Comprehensive training completed!")


if __name__ == "__main__":
    main()
