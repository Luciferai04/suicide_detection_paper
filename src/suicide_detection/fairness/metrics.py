#!/usr/bin/env python3
"""
Fairness metrics for suicide detection models.
Implements demographic parity, equal opportunity, and equalized odds.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, group_mask: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for a specific demographic group.

    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_prob: Prediction probabilities
        group_mask: Boolean mask for group membership

    Returns:
        Dictionary of metrics for the group
    """
    if np.sum(group_mask) == 0:
        return {}

    group_y_true = y_true[group_mask]
    group_y_pred = y_pred[group_mask]
    group_y_prob = y_prob[group_mask]

    # Handle edge cases where a group has only one class
    unique_classes = np.unique(group_y_true)

    metrics = {
        "n_samples": int(np.sum(group_mask)),
        "positive_rate": float(np.mean(group_y_pred)),
        "accuracy": float(accuracy_score(group_y_true, group_y_pred)),
    }

    if len(unique_classes) > 1:
        metrics.update(
            {
                "precision": float(precision_score(group_y_true, group_y_pred, zero_division=0)),
                "recall": float(recall_score(group_y_true, group_y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(group_y_true, group_y_prob)),
            }
        )

        # True positive and false positive rates
        tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
        metrics["tpr"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["tnr"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    else:
        # Single class metrics
        metrics.update(
            {
                "precision": 0.0,
                "recall": 0.0,
                "roc_auc": 0.5,
                "tpr": 0.0,
                "fpr": 0.0,
                "tnr": 1.0,
                "fnr": 1.0,
            }
        )

    return metrics


def demographic_parity(group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate demographic parity metrics.

    Demographic parity requires equal positive prediction rates across groups.

    Args:
        group_metrics: Dictionary mapping group names to their metrics

    Returns:
        Dictionary with demographic parity metrics
    """
    positive_rates = []
    for group, metrics in group_metrics.items():
        if "positive_rate" in metrics:
            positive_rates.append(metrics["positive_rate"])

    if len(positive_rates) < 2:
        return {"demographic_parity_difference": 0.0, "demographic_parity_ratio": 1.0}

    max_rate = max(positive_rates)
    min_rate = min(positive_rates)

    return {
        "demographic_parity_difference": float(max_rate - min_rate),
        "demographic_parity_ratio": float(min_rate / max_rate) if max_rate > 0 else 0.0,
    }


def equal_opportunity(group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate equal opportunity metrics.

    Equal opportunity requires equal true positive rates across groups.

    Args:
        group_metrics: Dictionary mapping group names to their metrics

    Returns:
        Dictionary with equal opportunity metrics
    """
    tpr_values = []
    for group, metrics in group_metrics.items():
        if "tpr" in metrics:
            tpr_values.append(metrics["tpr"])

    if len(tpr_values) < 2:
        return {"equal_opportunity_difference": 0.0, "equal_opportunity_ratio": 1.0}

    max_tpr = max(tpr_values)
    min_tpr = min(tpr_values)

    return {
        "equal_opportunity_difference": float(max_tpr - min_tpr),
        "equal_opportunity_ratio": float(min_tpr / max_tpr) if max_tpr > 0 else 0.0,
    }


def equalized_odds(group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate equalized odds metrics.

    Equalized odds requires equal TPR and FPR across groups.

    Args:
        group_metrics: Dictionary mapping group names to their metrics

    Returns:
        Dictionary with equalized odds metrics
    """
    tpr_values = []
    fpr_values = []

    for group, metrics in group_metrics.items():
        if "tpr" in metrics and "fpr" in metrics:
            tpr_values.append(metrics["tpr"])
            fpr_values.append(metrics["fpr"])

    if len(tpr_values) < 2:
        return {
            "equalized_odds_tpr_difference": 0.0,
            "equalized_odds_fpr_difference": 0.0,
            "equalized_odds_difference": 0.0,
        }

    tpr_diff = max(tpr_values) - min(tpr_values)
    fpr_diff = max(fpr_values) - min(fpr_values)

    return {
        "equalized_odds_tpr_difference": float(tpr_diff),
        "equalized_odds_fpr_difference": float(fpr_diff),
        "equalized_odds_difference": float(max(tpr_diff, fpr_diff)),
    }


def predictive_parity(group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate predictive parity metrics.

    Predictive parity requires equal precision across groups.

    Args:
        group_metrics: Dictionary mapping group names to their metrics

    Returns:
        Dictionary with predictive parity metrics
    """
    precision_values = []
    for group, metrics in group_metrics.items():
        if "precision" in metrics and metrics["precision"] > 0:
            precision_values.append(metrics["precision"])

    if len(precision_values) < 2:
        return {"predictive_parity_difference": 0.0, "predictive_parity_ratio": 1.0}

    max_prec = max(precision_values)
    min_prec = min(precision_values)

    return {
        "predictive_parity_difference": float(max_prec - min_prec),
        "predictive_parity_ratio": float(min_prec / max_prec) if max_prec > 0 else 0.0,
    }


def comprehensive_fairness_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    group_names: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Perform comprehensive fairness analysis across demographic groups.

    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_prob: Prediction probabilities
        groups: Group membership array
        group_names: Optional names for groups
        thresholds: Fairness metric thresholds for flagging issues

    Returns:
        Comprehensive fairness analysis results
    """
    # Default thresholds for fairness metrics
    if thresholds is None:
        thresholds = {
            "demographic_parity_difference": 0.1,
            "equal_opportunity_difference": 0.1,
            "equalized_odds_difference": 0.1,
            "predictive_parity_difference": 0.1,
        }

    unique_groups = np.unique(groups)
    if group_names is None:
        group_names = [str(g) for g in unique_groups]

    # Compute metrics for each group
    group_metrics = {}
    for group_val, group_name in zip(unique_groups, group_names):
        mask = groups == group_val
        if np.sum(mask) >= 10:  # Skip very small groups
            group_metrics[group_name] = compute_group_metrics(y_true, y_pred, y_prob, mask)

    # Calculate fairness metrics
    results = {
        "group_metrics": group_metrics,
        "fairness_metrics": {
            "demographic_parity": demographic_parity(group_metrics),
            "equal_opportunity": equal_opportunity(group_metrics),
            "equalized_odds": equalized_odds(group_metrics),
            "predictive_parity": predictive_parity(group_metrics),
        },
    }

    # Flag fairness issues
    issues = []
    for metric_type, metrics in results["fairness_metrics"].items():
        for metric_name, value in metrics.items():
            if "_difference" in metric_name:
                threshold = thresholds.get(metric_name, 0.1)
                if value > threshold:
                    issues.append(
                        {
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold,
                            "severity": "high" if value > threshold * 2 else "medium",
                        }
                    )

    results["fairness_issues"] = issues
    results["overall_fairness_score"] = calculate_overall_fairness_score(
        results["fairness_metrics"]
    )

    # Generate mitigation recommendations
    results["recommendations"] = generate_mitigation_recommendations(issues)

    return results


def calculate_overall_fairness_score(fairness_metrics: Dict[str, Dict[str, float]]) -> float:
    """Calculate an overall fairness score from 0 to 1.

    Higher scores indicate better fairness.

    Args:
        fairness_metrics: Dictionary of fairness metrics

    Returns:
        Overall fairness score
    """
    differences = []

    for metric_type, metrics in fairness_metrics.items():
        for metric_name, value in metrics.items():
            if "_difference" in metric_name:
                # Convert difference to a score (0 = worst, 1 = best)
                score = max(0, 1 - value)
                differences.append(score)

    if not differences:
        return 1.0

    # Return average of all fairness scores
    return float(np.mean(differences))


def generate_mitigation_recommendations(issues: List[Dict[str, Any]]) -> List[str]:
    """Generate mitigation recommendations based on fairness issues.

    Args:
        issues: List of identified fairness issues

    Returns:
        List of mitigation recommendations
    """
    recommendations = []

    # Check for specific types of issues
    has_dp_issue = any(i["metric"] == "demographic_parity_difference" for i in issues)
    has_eo_issue = any(i["metric"] == "equal_opportunity_difference" for i in issues)
    has_eod_issue = any(i["metric"] == "equalized_odds_difference" for i in issues)
    has_pp_issue = any(i["metric"] == "predictive_parity_difference" for i in issues)

    if has_dp_issue:
        recommendations.append(
            "Consider reweighting training samples to balance positive prediction rates across groups."
        )
        recommendations.append(
            "Implement threshold optimization per group to achieve demographic parity."
        )

    if has_eo_issue:
        recommendations.append(
            "Focus on improving recall for underperforming groups through targeted data augmentation."
        )
        recommendations.append(
            "Consider using fairness-aware loss functions that penalize TPR disparities."
        )

    if has_eod_issue:
        recommendations.append(
            "Apply post-processing calibration to equalize both TPR and FPR across groups."
        )
        recommendations.append(
            "Investigate potential label bias in the training data for different groups."
        )

    if has_pp_issue:
        recommendations.append(
            "Review precision disparities and consider group-specific confidence thresholds."
        )
        recommendations.append(
            "Ensure balanced representation of positive examples across all groups."
        )

    # General recommendations
    if issues:
        recommendations.append("Consider collecting more diverse and representative training data.")
        recommendations.append(
            "Implement regular fairness audits throughout the model development lifecycle."
        )
        recommendations.append(
            "Document all fairness considerations and trade-offs for stakeholders."
        )

    return recommendations


def save_fairness_report(results: Dict[str, Any], output_path: Path, format: str = "both") -> None:
    """Save fairness analysis results to file.

    Args:
        results: Fairness analysis results
        output_path: Base path for output files
        format: Output format ('json', 'markdown', or 'both')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved fairness JSON report to {json_path}")

    if format in ["markdown", "both"]:
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write(generate_markdown_report(results))
        logger.info(f"Saved fairness Markdown report to {md_path}")


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate a Markdown report from fairness results.

    Args:
        results: Fairness analysis results

    Returns:
        Markdown formatted report
    """
    lines = ["# Fairness Analysis Report\n"]

    # Overall score
    lines.append(f"## Overall Fairness Score: {results['overall_fairness_score']:.3f}\n")

    # Group metrics
    lines.append("## Group-wise Performance Metrics\n")
    for group, metrics in results["group_metrics"].items():
        lines.append(f"\n### Group: {group}")
        lines.append(f"- Sample size: {metrics.get('n_samples', 'N/A')}")
        lines.append(f"- Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
        lines.append(f"- Precision: {metrics.get('precision', 'N/A'):.3f}")
        lines.append(f"- Recall: {metrics.get('recall', 'N/A'):.3f}")
        lines.append(f"- ROC-AUC: {metrics.get('roc_auc', 'N/A'):.3f}")

    # Fairness metrics
    lines.append("\n## Fairness Metrics\n")
    for metric_type, metrics in results["fairness_metrics"].items():
        lines.append(f"\n### {metric_type.replace('_', ' ').title()}")
        for name, value in metrics.items():
            lines.append(f"- {name.replace('_', ' ').title()}: {value:.3f}")

    # Issues
    if results["fairness_issues"]:
        lines.append("\n## ⚠️ Fairness Issues Detected\n")
        for issue in results["fairness_issues"]:
            severity_icon = "🔴" if issue["severity"] == "high" else "🟡"
            lines.append(
                f"{severity_icon} **{issue['metric']}**: "
                f"{issue['value']:.3f} (threshold: {issue['threshold']:.3f})"
            )
    else:
        lines.append("\n## ✅ No Significant Fairness Issues Detected\n")

    # Recommendations
    if results["recommendations"]:
        lines.append("\n## Recommendations\n")
        for rec in results["recommendations"]:
            lines.append(f"- {rec}")

    return "\n".join(lines)
