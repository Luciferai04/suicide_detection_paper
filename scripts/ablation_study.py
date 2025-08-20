#!/usr/bin/env python3
"""
Ablation Study Framework for Suicide Detection Models.
Tests the impact of different model components and configurations.
"""

import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class AblationStudy:
    """Manages ablation studies for suicide detection models."""
    
    def __init__(self, base_config_path: str = "configs/default.yaml"):
        self.base_config_path = base_config_path
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"results/ablation_studies/{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def define_ablation_experiments(self) -> List[Dict]:
        """Define all ablation experiments to run."""
        
        experiments = [
            # BERT Ablations
            {
                "name": "bert_base",
                "description": "BERT baseline (full model)",
                "model": "bert",
                "config": {
                    "bert": {
                        "model_name": "bert-base-uncased",
                        "max_length": 512,
                        "batch_size": 16,
                        "num_epochs": 3,
                        "learning_rate": 2e-5,
                        "warmup_ratio": 0.06
                    }
                }
            },
            {
                "name": "bert_no_warmup",
                "description": "BERT without learning rate warmup",
                "model": "bert",
                "config": {
                    "bert": {
                        "model_name": "bert-base-uncased",
                        "max_length": 512,
                        "batch_size": 16,
                        "num_epochs": 3,
                        "learning_rate": 2e-5,
                        "warmup_ratio": 0.0
                    }
                }
            },
            {
                "name": "bert_short_seq",
                "description": "BERT with shorter sequence length",
                "model": "bert",
                "config": {
                    "bert": {
                        "model_name": "bert-base-uncased",
                        "max_length": 256,
                        "batch_size": 32,
                        "num_epochs": 3,
                        "learning_rate": 2e-5,
                        "warmup_ratio": 0.06
                    }
                }
            },
            {
                "name": "bert_frozen_layers",
                "description": "BERT with frozen embedding layers",
                "model": "bert",
                "config": {
                    "bert": {
                        "model_name": "bert-base-uncased",
                        "max_length": 512,
                        "batch_size": 16,
                        "num_epochs": 3,
                        "learning_rate": 2e-5,
                        "warmup_ratio": 0.06,
                        "freeze_embeddings": True,
                        "freeze_layers": 8
                    }
                }
            },
            {
                "name": "roberta_base",
                "description": "RoBERTa instead of BERT",
                "model": "bert",
                "config": {
                    "bert": {
                        "model_name": "roberta-base",
                        "max_length": 512,
                        "batch_size": 16,
                        "num_epochs": 3,
                        "learning_rate": 2e-5,
                        "warmup_ratio": 0.06
                    }
                }
            },
            {
                "name": "distilbert",
                "description": "DistilBERT (smaller, faster)",
                "model": "bert",
                "config": {
                    "bert": {
                        "model_name": "distilbert-base-uncased",
                        "max_length": 512,
                        "batch_size": 32,
                        "num_epochs": 3,
                        "learning_rate": 2e-5,
                        "warmup_ratio": 0.06
                    }
                }
            },
            
            # BiLSTM Ablations
            {
                "name": "bilstm_base",
                "description": "BiLSTM baseline (with attention)",
                "model": "bilstm",
                "config": {
                    "bilstm": {
                        "embedding_dim": 300,
                        "hidden_dim": 256,
                        "num_layers": 2,
                        "dropout": 0.3,
                        "use_attention": True,
                        "batch_size": 64,
                        "num_epochs": 10,
                        "learning_rate": 0.001
                    }
                }
            },
            {
                "name": "bilstm_no_attention",
                "description": "BiLSTM without attention mechanism",
                "model": "bilstm",
                "config": {
                    "bilstm": {
                        "embedding_dim": 300,
                        "hidden_dim": 256,
                        "num_layers": 2,
                        "dropout": 0.3,
                        "use_attention": False,
                        "batch_size": 64,
                        "num_epochs": 10,
                        "learning_rate": 0.001
                    }
                }
            },
            {
                "name": "bilstm_single_layer",
                "description": "BiLSTM with single layer",
                "model": "bilstm",
                "config": {
                    "bilstm": {
                        "embedding_dim": 300,
                        "hidden_dim": 256,
                        "num_layers": 1,
                        "dropout": 0.3,
                        "use_attention": True,
                        "batch_size": 64,
                        "num_epochs": 10,
                        "learning_rate": 0.001
                    }
                }
            },
            {
                "name": "bilstm_small_hidden",
                "description": "BiLSTM with smaller hidden dimension",
                "model": "bilstm",
                "config": {
                    "bilstm": {
                        "embedding_dim": 300,
                        "hidden_dim": 128,
                        "num_layers": 2,
                        "dropout": 0.3,
                        "use_attention": True,
                        "batch_size": 64,
                        "num_epochs": 10,
                        "learning_rate": 0.001
                    }
                }
            },
            {
                "name": "bilstm_small_embedding",
                "description": "BiLSTM with smaller embedding dimension",
                "model": "bilstm",
                "config": {
                    "bilstm": {
                        "embedding_dim": 100,
                        "hidden_dim": 256,
                        "num_layers": 2,
                        "dropout": 0.3,
                        "use_attention": True,
                        "batch_size": 64,
                        "num_epochs": 10,
                        "learning_rate": 0.001
                    }
                }
            },
            {
                "name": "bilstm_no_dropout",
                "description": "BiLSTM without dropout",
                "model": "bilstm",
                "config": {
                    "bilstm": {
                        "embedding_dim": 300,
                        "hidden_dim": 256,
                        "num_layers": 2,
                        "dropout": 0.0,
                        "use_attention": True,
                        "batch_size": 64,
                        "num_epochs": 10,
                        "learning_rate": 0.001
                    }
                }
            },
            {
                "name": "lstm_unidirectional",
                "description": "Unidirectional LSTM instead of BiLSTM",
                "model": "bilstm",
                "config": {
                    "bilstm": {
                        "embedding_dim": 300,
                        "hidden_dim": 256,
                        "num_layers": 2,
                        "dropout": 0.3,
                        "use_attention": True,
                        "bidirectional": False,
                        "batch_size": 64,
                        "num_epochs": 10,
                        "learning_rate": 0.001
                    }
                }
            },
            
            # SVM Ablations
            {
                "name": "svm_base",
                "description": "SVM baseline (TF-IDF + linear kernel)",
                "model": "svm",
                "config": {
                    "svm": {
                        "kernel": "linear",
                        "C": 1.0,
                        "max_features": 10000,
                        "ngram_range": [1, 2],
                        "use_tfidf": True
                    }
                }
            },
            {
                "name": "svm_rbf",
                "description": "SVM with RBF kernel",
                "model": "svm",
                "config": {
                    "svm": {
                        "kernel": "rbf",
                        "C": 1.0,
                        "gamma": "scale",
                        "max_features": 10000,
                        "ngram_range": [1, 2],
                        "use_tfidf": True
                    }
                }
            },
            {
                "name": "svm_unigram",
                "description": "SVM with unigrams only",
                "model": "svm",
                "config": {
                    "svm": {
                        "kernel": "linear",
                        "C": 1.0,
                        "max_features": 10000,
                        "ngram_range": [1, 1],
                        "use_tfidf": True
                    }
                }
            },
            {
                "name": "svm_trigram",
                "description": "SVM with up to trigrams",
                "model": "svm",
                "config": {
                    "svm": {
                        "kernel": "linear",
                        "C": 1.0,
                        "max_features": 10000,
                        "ngram_range": [1, 3],
                        "use_tfidf": True
                    }
                }
            },
            {
                "name": "svm_bow",
                "description": "SVM with Bag-of-Words (no TF-IDF)",
                "model": "svm",
                "config": {
                    "svm": {
                        "kernel": "linear",
                        "C": 1.0,
                        "max_features": 10000,
                        "ngram_range": [1, 2],
                        "use_tfidf": False
                    }
                }
            },
            {
                "name": "svm_small_vocab",
                "description": "SVM with smaller vocabulary",
                "model": "svm",
                "config": {
                    "svm": {
                        "kernel": "linear",
                        "C": 1.0,
                        "max_features": 5000,
                        "ngram_range": [1, 2],
                        "use_tfidf": True
                    }
                }
            }
        ]
        
        return experiments
    
    def run_experiment(self, experiment: Dict) -> Dict:
        """Run a single ablation experiment."""
        
        print(f"\n{'='*80}")
        print(f"Running: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"{'='*80}")
        
        # Create config file for this experiment
        config_file = self.output_dir / f"config_{experiment['name']}.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(experiment['config'], f)
        
        # Prepare output directory
        output_dir = self.output_dir / experiment['name']
        output_dir.mkdir(exist_ok=True)
        
        # Run training
        cmd = [
            "python", "-m", "suicide_detection.training.train",
            "--model", experiment['model'],
            "--dataset", "kaggle",
            "--config", str(config_file),
            "--output_dir", str(output_dir),
            "--prefer_device", "cpu",
            "--quick_test"  # Use smaller subset for faster ablation
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            training_time = time.time() - start_time
            
            # Load metrics
            metrics_file = output_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}
            
            # Store result
            result_dict = {
                "experiment": experiment['name'],
                "description": experiment['description'],
                "model": experiment['model'],
                "config": experiment['config'],
                "metrics": metrics,
                "training_time": training_time,
                "success": result.returncode == 0,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result_dict)
            
            print(f"✅ Completed: {experiment['name']}")
            if metrics:
                print(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
                print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout: {experiment['name']}")
            return {
                "experiment": experiment['name'],
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            print(f"❌ Error: {experiment['name']} - {str(e)}")
            return {
                "experiment": experiment['name'],
                "success": False,
                "error": str(e)
            }
    
    def analyze_results(self) -> pd.DataFrame:
        """Analyze ablation study results."""
        
        # Convert to DataFrame
        df_data = []
        for result in self.results:
            if result.get('success', False) and 'metrics' in result:
                row = {
                    'Experiment': result['experiment'],
                    'Description': result['description'],
                    'Model': result['model'],
                    'Accuracy': result['metrics'].get('accuracy', 0),
                    'Precision': result['metrics'].get('precision', 0),
                    'Recall': result['metrics'].get('recall', 0),
                    'F1-Score': result['metrics'].get('f1_score', 0),
                    'AUC-ROC': result['metrics'].get('roc_auc', 0),
                    'Training Time (s)': result.get('training_time', 0)
                }
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Group by model type
        model_groups = df.groupby('Model')
        
        # Calculate impact of each ablation
        impact_analysis = []
        for model, group in model_groups:
            baseline_name = f"{model}_base"
            baseline = group[group['Experiment'] == baseline_name]
            
            if not baseline.empty:
                baseline_f1 = baseline['F1-Score'].values[0]
                
                for _, row in group.iterrows():
                    if row['Experiment'] != baseline_name:
                        impact = row['F1-Score'] - baseline_f1
                        impact_pct = (impact / baseline_f1) * 100 if baseline_f1 > 0 else 0
                        
                        impact_analysis.append({
                            'Model': model,
                            'Ablation': row['Description'],
                            'Baseline F1': baseline_f1,
                            'Ablation F1': row['F1-Score'],
                            'Impact': impact,
                            'Impact (%)': impact_pct
                        })
        
        impact_df = pd.DataFrame(impact_analysis)
        
        return df, impact_df
    
    def plot_ablation_results(self):
        """Create visualizations of ablation study results."""
        
        df, impact_df = self.analyze_results()
        
        if df.empty:
            print("No results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. F1-Score comparison by model
        ax = axes[0, 0]
        for model in df['Model'].unique():
            model_df = df[df['Model'] == model].sort_values('F1-Score')
            ax.barh(model_df['Experiment'], model_df['F1-Score'], label=model, alpha=0.7)
        ax.set_xlabel('F1-Score')
        ax.set_title('F1-Score by Experiment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Impact analysis
        ax = axes[0, 1]
        if not impact_df.empty:
            impact_sorted = impact_df.sort_values('Impact (%)')
            colors = ['red' if x < 0 else 'green' for x in impact_sorted['Impact (%)']]
            ax.barh(range(len(impact_sorted)), impact_sorted['Impact (%)'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(impact_sorted)))
            ax.set_yticklabels([f"{a[:30]}..." if len(a) > 30 else a 
                                for a in impact_sorted['Ablation']], fontsize=8)
            ax.set_xlabel('Impact on F1-Score (%)')
            ax.set_title('Component Impact Analysis')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
        
        # 3. Training time comparison
        ax = axes[0, 2]
        time_df = df.sort_values('Training Time (s)')
        ax.barh(time_df['Experiment'], time_df['Training Time (s)'], alpha=0.7)
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Training Efficiency')
        ax.grid(True, alpha=0.3)
        
        # 4. Metric correlation heatmap
        ax = axes[1, 0]
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        if all(col in df.columns for col in metrics_cols):
            corr_matrix = df[metrics_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Metric Correlations')
        
        # 5. Model-wise performance
        ax = axes[1, 1]
        model_perf = df.groupby('Model')[['F1-Score', 'Accuracy', 'AUC-ROC']].mean()
        model_perf.plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Performance by Model Type')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # 6. Top ablations
        ax = axes[1, 2]
        top_ablations = df.nlargest(10, 'F1-Score')[['Experiment', 'F1-Score', 'AUC-ROC']]
        x = range(len(top_ablations))
        width = 0.35
        ax.bar([i - width/2 for i in x], top_ablations['F1-Score'], width, label='F1-Score', alpha=0.7)
        ax.bar([i + width/2 for i in x], top_ablations['AUC-ROC'], width, label='AUC-ROC', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([e[:15] for e in top_ablations['Experiment']], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Score')
        ax.set_title('Top 10 Configurations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "ablation_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved ablation analysis plot to {plot_path}")
        
        return fig
    
    def save_results(self):
        """Save ablation study results."""
        
        # Save raw results
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save analysis
        df, impact_df = self.analyze_results()
        
        if not df.empty:
            df.to_csv(self.output_dir / "ablation_metrics.csv", index=False)
            
        if not impact_df.empty:
            impact_df.to_csv(self.output_dir / "ablation_impact.csv", index=False)
        
        # Generate report
        report_file = self.output_dir / "ablation_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ABLATION STUDY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Successful: {sum(1 for r in self.results if r.get('success', False))}\n\n")
            
            if not df.empty:
                f.write("TOP PERFORMING CONFIGURATIONS\n")
                f.write("-"*40 + "\n")
                top_5 = df.nlargest(5, 'F1-Score')
                for _, row in top_5.iterrows():
                    f.write(f"\n{row['Experiment']}\n")
                    f.write(f"  Description: {row['Description']}\n")
                    f.write(f"  F1-Score: {row['F1-Score']:.3f}\n")
                    f.write(f"  Accuracy: {row['Accuracy']:.3f}\n")
                    f.write(f"  AUC-ROC: {row['AUC-ROC']:.3f}\n")
            
            if not impact_df.empty:
                f.write("\n\nMOST IMPACTFUL COMPONENTS\n")
                f.write("-"*40 + "\n")
                
                # Positive impact
                positive = impact_df[impact_df['Impact (%)'] > 0].nlargest(5, 'Impact (%)')
                if not positive.empty:
                    f.write("\nPositive Impact:\n")
                    for _, row in positive.iterrows():
                        f.write(f"  {row['Ablation']}: +{row['Impact (%)']:.1f}%\n")
                
                # Negative impact
                negative = impact_df[impact_df['Impact (%)'] < 0].nsmallest(5, 'Impact (%)')
                if not negative.empty:
                    f.write("\nNegative Impact (components essential for performance):\n")
                    for _, row in negative.iterrows():
                        f.write(f"  {row['Ablation']}: {row['Impact (%)']:.1f}%\n")
        
        print(f"Saved ablation study results to {self.output_dir}")
    
    def run_full_study(self, experiments: Optional[List[str]] = None):
        """Run complete ablation study."""
        
        all_experiments = self.define_ablation_experiments()
        
        # Filter experiments if specified
        if experiments:
            all_experiments = [e for e in all_experiments if e['name'] in experiments]
        
        print(f"Running {len(all_experiments)} ablation experiments...")
        
        # Run experiments
        for experiment in tqdm(all_experiments, desc="Ablation experiments"):
            self.run_experiment(experiment)
        
        # Analyze and save results
        self.plot_ablation_results()
        self.save_results()
        
        print("\n✅ Ablation study complete!")

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of experiments")
    args = parser.parse_args()
    
    study = AblationStudy()
    
    if args.quick:
        # Quick subset for testing
        experiments = [
            "bert_base", "bert_no_warmup", "bert_short_seq",
            "bilstm_base", "bilstm_no_attention",
            "svm_base", "svm_unigram"
        ]
        study.run_full_study(experiments)
    else:
        study.run_full_study(args.experiments)

if __name__ == "__main__":
    main()
