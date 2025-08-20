#!/usr/bin/env python3
"""
Generate comparison charts and visualizations for training results.
Creates bar charts, line plots, and performance comparisons.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class ChartGenerator:
    def __init__(self):
        self.results_dir = Path("results")
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Color palette for devices
        self.device_colors = {
            'mps': '#FF6B6B',     # Red for Apple Silicon
            'cuda': '#4ECDC4',    # Teal for NVIDIA
            'cpu': '#95E77E'      # Green for CPU
        }
        
        self.model_markers = {
            'bert': 'o',
            'bilstm': 's',
            'svm': '^'
        }
    
    def load_latest_results(self) -> List[Dict]:
        """Load the most recent training results."""
        result_files = list(self.results_dir.glob("training_results_*.json"))
        if not result_files:
            print("No training results found")
            return []
        
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def create_accuracy_comparison(self, results: List[Dict]):
        """Create accuracy comparison bar chart."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        df_data = []
        for result in results:
            model_parts = result['model_name'].split('_')
            model_type = model_parts[0]
            device = result.get('device', 'cpu')
            
            df_data.append({
                'Model': model_type.upper(),
                'Device': device.upper(),
                'Accuracy': result.get('accuracy', 0),
                'Configuration': result['model_name']
            })
        
        df = pd.DataFrame(df_data)
        
        # Create grouped bar chart
        models = df['Model'].unique()
        devices = df['Device'].unique()
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, device in enumerate(devices):
            device_data = df[df['Device'] == device]
            accuracies = [device_data[device_data['Model'] == model]['Accuracy'].values[0] 
                         if len(device_data[device_data['Model'] == model]) > 0 else 0 
                         for model in models]
            
            color = self.device_colors.get(device.lower(), '#888888')
            bars = ax.bar(x + i * width, accuracies, width, 
                          label=device, color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                if acc > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison by Device', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend(title='Device', loc='upper left')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for baseline
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
        
        plt.tight_layout()
        save_path = self.plots_dir / f"accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Accuracy comparison chart saved to {save_path}")
        return save_path
    
    def create_performance_speedup_chart(self, results: List[Dict]):
        """Create performance speedup comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate speedups
        speedup_data = []
        
        # Group by model type
        models = {}
        for result in results:
            model_type = result['model_name'].split('_')[0]
            if model_type not in models:
                models[model_type] = {}
            device = result.get('device', 'cpu')
            models[model_type][device] = result.get('training_time', 0)
        
        # Calculate speedup relative to CPU
        for model_type, devices in models.items():
            if 'cpu' in devices and devices['cpu'] > 0:
                cpu_time = devices['cpu']
                for device, time in devices.items():
                    if time > 0:
                        speedup = cpu_time / time
                        speedup_data.append({
                            'Model': model_type.upper(),
                            'Device': device.upper(),
                            'Speedup': speedup
                        })
        
        if not speedup_data:
            print("⚠️  No training time data available for speedup calculation")
            return None
        
        df = pd.DataFrame(speedup_data)
        
        # Create bar chart
        pivot_df = df.pivot(index='Model', columns='Device', values='Speedup')
        pivot_df.plot(kind='bar', ax=ax, color=[self.device_colors.get(d.lower(), '#888888') 
                                                 for d in pivot_df.columns])
        
        ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (relative to CPU)', fontsize=12, fontweight='bold')
        ax.set_title('Training Speedup Comparison', fontsize=14, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title='Device')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2fx')
        
        plt.tight_layout()
        save_path = self.plots_dir / f"speedup_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Speedup comparison chart saved to {save_path}")
        return save_path
    
    def create_metrics_heatmap(self, results: List[Dict]):
        """Create a heatmap of all metrics."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data
        metrics_data = []
        for result in results:
            metrics_data.append({
                'Configuration': result['model_name'],
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1-Score': result.get('f1_score', 0)
            })
        
        df = pd.DataFrame(metrics_data)
        df = df.set_index('Configuration')
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        
        ax.set_title('Model Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Model Configuration', fontsize=12)
        
        plt.tight_layout()
        save_path = self.plots_dir / f"metrics_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Metrics heatmap saved to {save_path}")
        return save_path
    
    def create_device_comparison_radar(self, results: List[Dict]):
        """Create radar chart comparing devices across metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Group by model type
        model_types = ['bert', 'bilstm', 'svm']
        
        for idx, model_type in enumerate(model_types):
            ax = axes[idx]
            
            # Filter results for this model type
            model_results = [r for r in results if r['model_name'].startswith(model_type)]
            
            if not model_results:
                continue
            
            # Prepare data for each device
            for result in model_results:
                device = result.get('device', 'cpu')
                values = [
                    result.get('accuracy', 0),
                    result.get('precision', 0),
                    result.get('recall', 0),
                    result.get('f1_score', 0)
                ]
                
                # Number of variables
                num_vars = len(metrics)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                values += values[:1]  # Complete the circle
                angles += angles[:1]
                
                # Plot
                color = self.device_colors.get(device, '#888888')
                ax.plot(angles, values, 'o-', linewidth=2, label=device.upper(), color=color)
                ax.fill(angles, values, alpha=0.25, color=color)
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title(f'{model_type.upper()} Performance', fontsize=12, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
        
        plt.suptitle('Device Performance Comparison (Radar Charts)', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        save_path = self.plots_dir / f"radar_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Radar comparison chart saved to {save_path}")
        return save_path
    
    def create_training_time_comparison(self, results: List[Dict]):
        """Create training time comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        time_data = []
        for result in results:
            if 'training_time' in result:
                model_type = result['model_name'].split('_')[0]
                device = result.get('device', 'cpu')
                time_data.append({
                    'Model': model_type.upper(),
                    'Device': device.upper(),
                    'Time (minutes)': result['training_time'] / 60,
                    'Configuration': result['model_name']
                })
        
        if not time_data:
            print("⚠️  No training time data available")
            return None
        
        df = pd.DataFrame(time_data)
        
        # Create grouped bar chart
        models = df['Model'].unique()
        devices = df['Device'].unique()
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, device in enumerate(devices):
            device_data = df[df['Device'] == device]
            times = [device_data[device_data['Model'] == model]['Time (minutes)'].values[0] 
                    if len(device_data[device_data['Model'] == model]) > 0 else 0 
                    for model in models]
            
            color = self.device_colors.get(device.lower(), '#888888')
            bars = ax.bar(x + i * width, times, width, 
                         label=device, color=color, alpha=0.8)
            
            # Add value labels
            for bar, time in zip(bars, times):
                if time > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{time:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time Comparison by Device', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend(title='Device')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.plots_dir / f"training_time_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training time comparison chart saved to {save_path}")
        return save_path
    
    def generate_all_charts(self):
        """Generate all comparison charts."""
        print("📊 Generating comparison charts...")
        
        # Load results
        results = self.load_latest_results()
        if not results:
            print("❌ No results to visualize")
            return
        
        print(f"Loaded {len(results)} training results")
        
        # Generate charts
        chart_paths = []
        
        # 1. Accuracy comparison
        path = self.create_accuracy_comparison(results)
        if path:
            chart_paths.append(path)
        
        # 2. Performance speedup
        path = self.create_performance_speedup_chart(results)
        if path:
            chart_paths.append(path)
        
        # 3. Metrics heatmap
        path = self.create_metrics_heatmap(results)
        if path:
            chart_paths.append(path)
        
        # 4. Radar charts
        path = self.create_device_comparison_radar(results)
        if path:
            chart_paths.append(path)
        
        # 5. Training time comparison
        path = self.create_training_time_comparison(results)
        if path:
            chart_paths.append(path)
        
        # Save chart paths for reference
        chart_manifest = self.plots_dir / "chart_manifest.json"
        with open(chart_manifest, 'w') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'charts': [str(p) for p in chart_paths]
            }, f, indent=2)
        
        print(f"\n✅ Generated {len(chart_paths)} charts")
        print(f"📁 Charts saved to: {self.plots_dir}")
        
        return chart_paths

def main():
    generator = ChartGenerator()
    generator.generate_all_charts()

if __name__ == "__main__":
    main()
