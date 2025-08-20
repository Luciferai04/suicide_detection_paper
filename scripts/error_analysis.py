#!/usr/bin/env python3
"""
Comprehensive Error Analysis Framework for Suicide Detection Models.
Analyzes misclassifications, categorizes errors, and identifies failure modes.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import re
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class ErrorAnalyzer:
    """Comprehensive error analysis for suicide detection models."""
    
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_dir = Path(f"results/error_analysis/{model_name}_{self.timestamp}")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Error categories
        self.error_categories = {
            "false_positive": {
                "ambiguous_language": "Text with unclear or ambiguous suicide-related language",
                "metaphorical": "Metaphorical or figurative language misinterpreted",
                "contextual": "Requires broader context to understand correctly",
                "sarcasm": "Sarcastic or ironic statements",
                "support_seeking": "Support-seeking messages misclassified as risk"
            },
            "false_negative": {
                "subtle_indicators": "Subtle or indirect risk indicators",
                "euphemistic": "Euphemistic language about suicide",
                "plan_discussion": "Discussion of specific plans or methods",
                "temporal": "Temporal progression not captured",
                "coded_language": "Use of coded or community-specific language"
            }
        }
        
    def load_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load model predictions and ground truth."""
        
        # Try to load from model output directory
        pred_file = self.output_dir / "test_predictions.npy"
        true_file = self.output_dir / "test_labels.npy"
        prob_file = self.output_dir / "test_probabilities.npy"
        text_file = self.output_dir / "test_texts.json"
        
        if not all([pred_file.exists(), true_file.exists()]):
            # Try alternative locations
            pred_file = self.output_dir / "y_pred_test.npy"
            true_file = self.output_dir / "y_true_test.npy"
            prob_file = self.output_dir / "y_prob_test.npy"
        
        y_pred = np.load(pred_file) if pred_file.exists() else None
        y_true = np.load(true_file) if true_file.exists() else None
        y_prob = np.load(prob_file) if prob_file.exists() else None
        
        # Load text data if available
        texts = []
        if text_file.exists():
            with open(text_file, 'r') as f:
                texts = json.load(f)
        
        return y_true, y_pred, y_prob, texts
    
    def categorize_errors(self, texts: List[str], y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict:
        """Categorize errors into different types."""
        
        error_analysis = {
            "false_positives": [],
            "false_negatives": [],
            "true_positives": [],
            "true_negatives": []
        }
        
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            text = texts[i] if i < len(texts) else ""
            
            if true_label == 1 and pred_label == 1:
                error_analysis["true_positives"].append({
                    "index": i,
                    "text": text,
                    "confidence": None
                })
            elif true_label == 0 and pred_label == 0:
                error_analysis["true_negatives"].append({
                    "index": i,
                    "text": text,
                    "confidence": None
                })
            elif true_label == 0 and pred_label == 1:
                # False positive
                category = self._categorize_false_positive(text)
                error_analysis["false_positives"].append({
                    "index": i,
                    "text": text,
                    "category": category,
                    "confidence": None
                })
            else:  # true_label == 1 and pred_label == 0
                # False negative
                category = self._categorize_false_negative(text)
                error_analysis["false_negatives"].append({
                    "index": i,
                    "text": text,
                    "category": category,
                    "confidence": None
                })
        
        return error_analysis
    
    def _categorize_false_positive(self, text: str) -> str:
        """Categorize false positive errors."""
        
        text_lower = text.lower()
        
        # Check for sarcasm indicators
        if any(indicator in text_lower for indicator in ["lol", "haha", "jk", "/s", "😂", "🤣"]):
            return "sarcasm"
        
        # Check for metaphorical language
        if any(word in text_lower for word in ["dying to", "kill for", "dead tired", "suicide mission"]):
            return "metaphorical"
        
        # Check for support seeking
        if any(phrase in text_lower for phrase in ["how to help", "worried about", "friend is", "someone I know"]):
            return "support_seeking"
        
        # Check for ambiguous language
        if len(text.split()) < 10:
            return "ambiguous_language"
        
        return "contextual"
    
    def _categorize_false_negative(self, text: str) -> str:
        """Categorize false negative errors."""
        
        text_lower = text.lower()
        
        # Check for euphemistic language
        euphemisms = ["end it all", "not be here", "gone", "leave this world", "no longer suffering"]
        if any(phrase in text_lower for phrase in euphemisms):
            return "euphemistic"
        
        # Check for plan discussion
        methods = ["pills", "bridge", "gun", "rope", "overdose", "jump"]
        if any(method in text_lower for method in methods):
            return "plan_discussion"
        
        # Check for coded language
        coded_terms = ["ctb", "catch the bus", "sui", "kms"]
        if any(term in text_lower for term in coded_terms):
            return "coded_language"
        
        # Check text length for subtle indicators
        if len(text.split()) > 100:
            return "subtle_indicators"
        
        return "temporal"
    
    def analyze_confidence_distribution(self, y_prob: np.ndarray, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> Dict:
        """Analyze confidence distribution for correct and incorrect predictions."""
        
        correct_mask = (y_true == y_pred)
        incorrect_mask = ~correct_mask
        
        analysis = {
            "correct_predictions": {
                "mean_confidence": np.mean(y_prob[correct_mask]) if np.any(correct_mask) else 0,
                "std_confidence": np.std(y_prob[correct_mask]) if np.any(correct_mask) else 0,
                "min_confidence": np.min(y_prob[correct_mask]) if np.any(correct_mask) else 0,
                "max_confidence": np.max(y_prob[correct_mask]) if np.any(correct_mask) else 0
            },
            "incorrect_predictions": {
                "mean_confidence": np.mean(y_prob[incorrect_mask]) if np.any(incorrect_mask) else 0,
                "std_confidence": np.std(y_prob[incorrect_mask]) if np.any(incorrect_mask) else 0,
                "min_confidence": np.min(y_prob[incorrect_mask]) if np.any(incorrect_mask) else 0,
                "max_confidence": np.max(y_prob[incorrect_mask]) if np.any(incorrect_mask) else 0
            },
            "calibration": self._calculate_calibration(y_prob, y_true)
        }
        
        return analysis
    
    def _calculate_calibration(self, y_prob: np.ndarray, y_true: np.ndarray, 
                               n_bins: int = 10) -> Dict:
        """Calculate calibration metrics."""
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_data = []
        
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if np.any(mask):
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_prob[mask])
                bin_count = np.sum(mask)
                
                calibration_data.append({
                    "bin_range": (bin_edges[i], bin_edges[i + 1]),
                    "bin_accuracy": bin_accuracy,
                    "bin_confidence": bin_confidence,
                    "bin_count": bin_count,
                    "calibration_error": abs(bin_accuracy - bin_confidence)
                })
        
        # Calculate Expected Calibration Error (ECE)
        total_samples = len(y_true)
        ece = sum(d["bin_count"] / total_samples * d["calibration_error"] 
                 for d in calibration_data)
        
        return {
            "bins": calibration_data,
            "expected_calibration_error": ece
        }
    
    def analyze_text_patterns(self, texts: List[str], y_true: np.ndarray, 
                             y_pred: np.ndarray) -> Dict:
        """Analyze text patterns in errors."""
        
        # Separate texts by prediction outcome
        fp_texts = [texts[i] for i in range(len(texts)) 
                   if i < len(y_true) and y_true[i] == 0 and y_pred[i] == 1]
        fn_texts = [texts[i] for i in range(len(texts)) 
                   if i < len(y_true) and y_true[i] == 1 and y_pred[i] == 0]
        
        analysis = {
            "false_positives": self._analyze_text_group(fp_texts),
            "false_negatives": self._analyze_text_group(fn_texts)
        }
        
        return analysis
    
    def _analyze_text_group(self, texts: List[str]) -> Dict:
        """Analyze patterns in a group of texts."""
        
        if not texts:
            return {}
        
        # Length analysis
        lengths = [len(text.split()) for text in texts]
        
        # Common words (excluding stopwords)
        all_words = ' '.join(texts).lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                    'are', 'were', 'been', 'be', 'i', 'me', 'my', 'we', 'you', 'it'}
        filtered_words = [w for w in all_words if w not in stopwords and len(w) > 2]
        word_freq = Counter(filtered_words)
        
        # N-gram analysis
        bigrams = []
        for text in texts:
            words = text.lower().split()
            bigrams.extend([f"{words[i]} {words[i+1]}" 
                          for i in range(len(words)-1)])
        bigram_freq = Counter(bigrams)
        
        return {
            "count": len(texts),
            "avg_length": np.mean(lengths) if lengths else 0,
            "std_length": np.std(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "top_words": dict(word_freq.most_common(20)),
            "top_bigrams": dict(bigram_freq.most_common(15))
        }
    
    def create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray], error_analysis: Dict):
        """Create comprehensive error analysis visualizations."""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 4, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['No Risk', 'Risk'],
                   yticklabels=['No Risk', 'Risk'])
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. Error Distribution
        ax2 = plt.subplot(3, 4, 2)
        error_counts = {
            'True Positive': len(error_analysis['true_positives']),
            'True Negative': len(error_analysis['true_negatives']),
            'False Positive': len(error_analysis['false_positives']),
            'False Negative': len(error_analysis['false_negatives'])
        }
        colors = ['green', 'lightgreen', 'orange', 'red']
        ax2.bar(error_counts.keys(), error_counts.values(), color=colors, alpha=0.7)
        ax2.set_title('Prediction Distribution', fontweight='bold')
        ax2.set_ylabel('Count')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Confidence Distribution
        if y_prob is not None:
            ax3 = plt.subplot(3, 4, 3)
            correct_mask = (y_true == y_pred)
            ax3.hist([y_prob[correct_mask], y_prob[~correct_mask]], 
                    bins=20, alpha=0.7, label=['Correct', 'Incorrect'])
            ax3.set_title('Confidence Distribution', fontweight='bold')
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('Count')
            ax3.legend()
            
            # 4. Calibration Plot
            ax4 = plt.subplot(3, 4, 4)
            calibration = self._calculate_calibration(y_prob, y_true)
            bin_confidences = [b['bin_confidence'] for b in calibration['bins']]
            bin_accuracies = [b['bin_accuracy'] for b in calibration['bins']]
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            ax4.scatter(bin_confidences, bin_accuracies, s=100, alpha=0.7)
            ax4.plot(bin_confidences, bin_accuracies, 'b-', alpha=0.5)
            ax4.set_title(f'Calibration Plot (ECE={calibration["expected_calibration_error"]:.3f})', 
                         fontweight='bold')
            ax4.set_xlabel('Mean Predicted Probability')
            ax4.set_ylabel('Fraction of Positives')
            ax4.legend()
        
        # 5. Error Category Distribution (False Positives)
        ax5 = plt.subplot(3, 4, 5)
        fp_categories = Counter([e.get('category', 'unknown') 
                                for e in error_analysis['false_positives']])
        if fp_categories:
            ax5.pie(fp_categories.values(), labels=fp_categories.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax5.set_title('False Positive Categories', fontweight='bold')
        
        # 6. Error Category Distribution (False Negatives)
        ax6 = plt.subplot(3, 4, 6)
        fn_categories = Counter([e.get('category', 'unknown') 
                                for e in error_analysis['false_negatives']])
        if fn_categories:
            ax6.pie(fn_categories.values(), labels=fn_categories.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax6.set_title('False Negative Categories', fontweight='bold')
        
        # 7. Precision-Recall by Threshold
        if y_prob is not None:
            ax7 = plt.subplot(3, 4, 7)
            thresholds = np.linspace(0.1, 0.9, 9)
            precisions = []
            recalls = []
            for threshold in thresholds:
                y_pred_thresh = (y_prob >= threshold).astype(int)
                tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
                fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
                fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            ax7.plot(thresholds, precisions, 'b-', label='Precision', marker='o')
            ax7.plot(thresholds, recalls, 'r-', label='Recall', marker='s')
            ax7.set_title('Precision-Recall vs Threshold', fontweight='bold')
            ax7.set_xlabel('Threshold')
            ax7.set_ylabel('Score')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Error Rate by Confidence Bins
        if y_prob is not None:
            ax8 = plt.subplot(3, 4, 8)
            bins = np.linspace(0, 1, 11)
            error_rates = []
            bin_centers = []
            
            for i in range(len(bins) - 1):
                mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
                if np.any(mask):
                    error_rate = np.mean(y_true[mask] != y_pred[mask])
                    error_rates.append(error_rate)
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
            
            ax8.bar(bin_centers, error_rates, width=0.08, alpha=0.7)
            ax8.set_title('Error Rate by Confidence', fontweight='bold')
            ax8.set_xlabel('Confidence Bin')
            ax8.set_ylabel('Error Rate')
            ax8.grid(True, alpha=0.3)
        
        # 9. Class Distribution
        ax9 = plt.subplot(3, 4, 9)
        class_dist = {
            'Actual Positive': np.sum(y_true == 1),
            'Actual Negative': np.sum(y_true == 0),
            'Predicted Positive': np.sum(y_pred == 1),
            'Predicted Negative': np.sum(y_pred == 0)
        }
        x = np.arange(len(class_dist))
        ax9.bar(x, class_dist.values(), alpha=0.7)
        ax9.set_xticks(x)
        ax9.set_xticklabels(class_dist.keys(), rotation=45, ha='right')
        ax9.set_title('Class Distribution', fontweight='bold')
        ax9.set_ylabel('Count')
        
        # 10. Performance by Text Length (if texts available)
        if error_analysis['false_positives'] or error_analysis['false_negatives']:
            ax10 = plt.subplot(3, 4, 10)
            
            # Placeholder for text length analysis
            ax10.text(0.5, 0.5, 'Text Length Analysis\n(Requires text data)', 
                     ha='center', va='center', fontsize=12)
            ax10.set_title('Performance by Text Length', fontweight='bold')
            ax10.set_xlim(0, 1)
            ax10.set_ylim(0, 1)
        
        # 11. Top Error-Inducing Features
        ax11 = plt.subplot(3, 4, 11)
        ax11.text(0.5, 0.5, 'Top Error Features\n(Analysis of most common\nerror-inducing terms)', 
                 ha='center', va='center', fontsize=12)
        ax11.set_title('Error-Inducing Features', fontweight='bold')
        ax11.set_xlim(0, 1)
        ax11.set_ylim(0, 1)
        
        # 12. Summary Statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        summary_text = f"""
        Model: {self.model_name}
        
        Performance Metrics:
        • Accuracy: {accuracy:.3f}
        • Precision: {precision:.3f}
        • Recall: {recall:.3f}
        • F1-Score: {f1:.3f}
        
        Error Analysis:
        • False Positives: {len(error_analysis['false_positives'])}
        • False Negatives: {len(error_analysis['false_negatives'])}
        • Error Rate: {1 - accuracy:.3f}
        """
        
        if y_prob is not None:
            calibration = self._calculate_calibration(y_prob, y_true)
            summary_text += f"\n• ECE: {calibration['expected_calibration_error']:.3f}"
        
        ax12.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                 fontfamily='monospace')
        ax12.set_title('Summary Statistics', fontweight='bold')
        
        plt.suptitle(f'Error Analysis - {self.model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.analysis_dir / "error_analysis_visualization.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved error analysis visualization to {fig_path}")
        
        return fig
    
    def generate_report(self, error_analysis: Dict, confidence_analysis: Dict, 
                       text_analysis: Dict) -> str:
        """Generate comprehensive error analysis report."""
        
        report = f"""
================================================================================
ERROR ANALYSIS REPORT
Model: {self.model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. OVERVIEW
-----------
Total Predictions: {sum(len(v) for v in error_analysis.values())}
True Positives: {len(error_analysis['true_positives'])}
True Negatives: {len(error_analysis['true_negatives'])}
False Positives: {len(error_analysis['false_positives'])}
False Negatives: {len(error_analysis['false_negatives'])}

2. ERROR CATEGORIES
-------------------
"""
        
        # False Positive Categories
        if error_analysis['false_positives']:
            fp_cats = Counter([e.get('category', 'unknown') 
                             for e in error_analysis['false_positives']])
            report += "\nFalse Positive Categories:\n"
            for cat, count in fp_cats.most_common():
                pct = count / len(error_analysis['false_positives']) * 100
                report += f"  • {cat}: {count} ({pct:.1f}%)\n"
        
        # False Negative Categories
        if error_analysis['false_negatives']:
            fn_cats = Counter([e.get('category', 'unknown') 
                             for e in error_analysis['false_negatives']])
            report += "\nFalse Negative Categories:\n"
            for cat, count in fn_cats.most_common():
                pct = count / len(error_analysis['false_negatives']) * 100
                report += f"  • {cat}: {count} ({pct:.1f}%)\n"
        
        # Confidence Analysis
        if confidence_analysis:
            report += f"""

3. CONFIDENCE ANALYSIS
----------------------
Correct Predictions:
  • Mean Confidence: {confidence_analysis['correct_predictions']['mean_confidence']:.3f}
  • Std Confidence: {confidence_analysis['correct_predictions']['std_confidence']:.3f}
  • Range: [{confidence_analysis['correct_predictions']['min_confidence']:.3f}, 
           {confidence_analysis['correct_predictions']['max_confidence']:.3f}]

Incorrect Predictions:
  • Mean Confidence: {confidence_analysis['incorrect_predictions']['mean_confidence']:.3f}
  • Std Confidence: {confidence_analysis['incorrect_predictions']['std_confidence']:.3f}
  • Range: [{confidence_analysis['incorrect_predictions']['min_confidence']:.3f}, 
           {confidence_analysis['incorrect_predictions']['max_confidence']:.3f}]

Calibration:
  • Expected Calibration Error: {confidence_analysis['calibration']['expected_calibration_error']:.3f}
"""
        
        # Text Pattern Analysis
        if text_analysis:
            report += """

4. TEXT PATTERN ANALYSIS
------------------------
"""
            for error_type, analysis in text_analysis.items():
                if analysis:
                    report += f"\n{error_type.replace('_', ' ').title()}:\n"
                    report += f"  • Count: {analysis.get('count', 0)}\n"
                    report += f"  • Avg Length: {analysis.get('avg_length', 0):.1f} words\n"
                    report += f"  • Std Length: {analysis.get('std_length', 0):.1f} words\n"
                    
                    if 'top_words' in analysis and analysis['top_words']:
                        report += "  • Top Words: "
                        top_5_words = list(analysis['top_words'].items())[:5]
                        report += ", ".join([f"{w}({c})" for w, c in top_5_words]) + "\n"
        
        report += """

5. RECOMMENDATIONS
------------------
"""
        
        # Generate recommendations based on error analysis
        recommendations = self._generate_recommendations(error_analysis, confidence_analysis)
        for i, rec in enumerate(recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        report += """

================================================================================
"""
        
        return report
    
    def _generate_recommendations(self, error_analysis: Dict, 
                                 confidence_analysis: Dict) -> List[str]:
        """Generate recommendations based on error analysis."""
        
        recommendations = []
        
        # Check false positive patterns
        if error_analysis['false_positives']:
            fp_cats = Counter([e.get('category', 'unknown') 
                             for e in error_analysis['false_positives']])
            top_fp = fp_cats.most_common(1)[0] if fp_cats else None
            
            if top_fp:
                if top_fp[0] == 'sarcasm':
                    recommendations.append(
                        "Consider adding sarcasm detection module to reduce false positives"
                    )
                elif top_fp[0] == 'metaphorical':
                    recommendations.append(
                        "Implement metaphor detection to distinguish figurative language"
                    )
                elif top_fp[0] == 'support_seeking':
                    recommendations.append(
                        "Add classifier to identify support-seeking vs at-risk messages"
                    )
        
        # Check false negative patterns
        if error_analysis['false_negatives']:
            fn_cats = Counter([e.get('category', 'unknown') 
                             for e in error_analysis['false_negatives']])
            top_fn = fn_cats.most_common(1)[0] if fn_cats else None
            
            if top_fn:
                if top_fn[0] == 'euphemistic':
                    recommendations.append(
                        "Expand training data with euphemistic expressions of suicidal ideation"
                    )
                elif top_fn[0] == 'coded_language':
                    recommendations.append(
                        "Include community-specific coded language in training vocabulary"
                    )
                elif top_fn[0] == 'subtle_indicators':
                    recommendations.append(
                        "Enhance model's ability to detect subtle risk indicators"
                    )
        
        # Check calibration
        if confidence_analysis and 'calibration' in confidence_analysis:
            ece = confidence_analysis['calibration']['expected_calibration_error']
            if ece > 0.1:
                recommendations.append(
                    f"Model calibration is poor (ECE={ece:.3f}). Consider temperature scaling"
                )
        
        # Check confidence distribution
        if confidence_analysis:
            mean_conf_incorrect = confidence_analysis['incorrect_predictions']['mean_confidence']
            if mean_conf_incorrect > 0.7:
                recommendations.append(
                    "Model is overconfident on errors. Consider uncertainty estimation techniques"
                )
        
        # General recommendations
        total_errors = len(error_analysis['false_positives']) + len(error_analysis['false_negatives'])
        total_predictions = sum(len(v) for v in error_analysis.values())
        error_rate = total_errors / total_predictions if total_predictions > 0 else 0
        
        if error_rate > 0.2:
            recommendations.append(
                "High error rate detected. Consider ensemble methods or model architecture changes"
            )
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory. Consider A/B testing in production")
        
        return recommendations
    
    def save_analysis(self, error_analysis: Dict, confidence_analysis: Dict, 
                     text_analysis: Dict, report: str):
        """Save all analysis results."""
        
        # Save error analysis
        with open(self.analysis_dir / "error_analysis.json", 'w') as f:
            json.dump(error_analysis, f, indent=2, default=str)
        
        # Save confidence analysis
        if confidence_analysis:
            with open(self.analysis_dir / "confidence_analysis.json", 'w') as f:
                json.dump(confidence_analysis, f, indent=2)
        
        # Save text analysis
        if text_analysis:
            with open(self.analysis_dir / "text_pattern_analysis.json", 'w') as f:
                json.dump(text_analysis, f, indent=2)
        
        # Save report
        with open(self.analysis_dir / "error_analysis_report.txt", 'w') as f:
            f.write(report)
        
        print(f"Saved all error analysis results to {self.analysis_dir}")
    
    def run_complete_analysis(self):
        """Run complete error analysis pipeline."""
        
        print(f"\n{'='*80}")
        print(f"Running Error Analysis for {self.model_name}")
        print(f"{'='*80}\n")
        
        # Load predictions
        y_true, y_pred, y_prob, texts = self.load_predictions()
        
        if y_true is None or y_pred is None:
            print("❌ Could not load predictions. Generating sample data for demonstration...")
            # Generate sample data for demonstration
            np.random.seed(42)
            n_samples = 1000
            y_true = np.random.binomial(1, 0.3, n_samples)
            y_pred = y_true.copy()
            # Add some errors
            error_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
            y_pred[error_indices] = 1 - y_pred[error_indices]
            y_prob = np.random.beta(2, 2, n_samples)
            texts = [f"Sample text {i}" for i in range(n_samples)]
        
        # Categorize errors
        print("Categorizing errors...")
        error_analysis = self.categorize_errors(texts, y_true, y_pred)
        
        # Confidence analysis
        confidence_analysis = {}
        if y_prob is not None:
            print("Analyzing confidence distribution...")
            confidence_analysis = self.analyze_confidence_distribution(y_prob, y_true, y_pred)
        
        # Text pattern analysis
        text_analysis = {}
        if texts:
            print("Analyzing text patterns...")
            text_analysis = self.analyze_text_patterns(texts, y_true, y_pred)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations(y_true, y_pred, y_prob, error_analysis)
        
        # Generate report
        print("Generating report...")
        report = self.generate_report(error_analysis, confidence_analysis, text_analysis)
        print(report)
        
        # Save results
        print("Saving analysis results...")
        self.save_analysis(error_analysis, confidence_analysis, text_analysis, report)
        
        print(f"\n✅ Error analysis complete! Results saved to {self.analysis_dir}")
        
        return error_analysis, confidence_analysis, text_analysis

def main():
    """Run error analysis for all available models."""
    
    # Find model output directories
    output_base = Path("results/model_outputs")
    model_dirs = [d for d in output_base.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("No model outputs found. Running demonstration with sample data...")
        analyzer = ErrorAnalyzer("demo_model", "results/model_outputs/demo")
        analyzer.run_complete_analysis()
    else:
        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"\nAnalyzing {model_name}...")
            
            analyzer = ErrorAnalyzer(model_name, str(model_dir))
            analyzer.run_complete_analysis()
    
    print("\n✅ All error analyses complete!")

if __name__ == "__main__":
    main()
