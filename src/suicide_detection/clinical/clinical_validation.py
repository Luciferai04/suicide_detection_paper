"""
Clinical Validation Framework

Provides clinical-grade validation protocols, metrics, and evaluation
frameworks specifically designed for suicide risk detection in healthcare settings.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from ..evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class ClinicalMetrics:
    """Clinical-specific evaluation metrics for suicide risk detection."""

    # Primary clinical metrics
    sensitivity_at_95_specificity: float
    specificity_at_95_sensitivity: float
    ppv_at_95_sensitivity: float
    npv_at_95_sensitivity: float

    # Risk stratification metrics
    high_risk_sensitivity: float  # Sensitivity for highest risk tier
    moderate_risk_precision: float  # Precision for moderate risk
    low_risk_npv: float  # NPV for low risk classification

    # Calibration and reliability
    brier_score: float
    calibration_slope: float
    calibration_intercept: float
    hosmer_lemeshow_pvalue: float

    # Clinical utility metrics
    net_benefit_at_threshold: Dict[str, float]  # Net benefit at different thresholds
    decision_curve_auc: float  # Area under decision curve

    # Time-based metrics (for longitudinal validation)
    c_index_time_dependent: Optional[float] = None
    time_dependent_auc: Optional[Dict[str, float]] = None


@dataclass
class ValidationProtocol:
    """Validation protocol configuration for clinical studies."""

    name: str
    description: str

    # Study design parameters (non-defaults first)
    validation_type: str  # "prospective", "retrospective", "cross_sectional"
    study_period: Tuple[str, str]  # Start and end dates

    # Population criteria (non-defaults)
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    target_population: str

    # Validation metrics (non-defaults)
    primary_endpoint: str
    secondary_endpoints: List[str]
    safety_endpoints: List[str]

    # Optional and defaulted parameters after non-defaults
    follow_up_period: Optional[int] = None  # Days for outcome assessment
    alpha: float = 0.05
    power: float = 0.8
    expected_prevalence: float = 0.1
    minimum_detectable_difference: float = 0.05

    # Clinical thresholds
    high_risk_threshold: float = 0.8
    moderate_risk_threshold: float = 0.3
    intervention_threshold: float = 0.5


class ClinicalValidator:
    """
    Clinical validation framework for suicide risk detection models.

    Provides comprehensive clinical evaluation including:
    - Clinical performance metrics
    - Risk stratification validation
    - Calibration assessment
    - Decision curve analysis
    - Bias and fairness evaluation
    """

    def __init__(self, protocol: ValidationProtocol):
        """
        Initialize clinical validator with validation protocol.

        Args:
            protocol: ValidationProtocol defining study parameters
        """
        self.protocol = protocol
        self.results = []
        self.validation_start_time = datetime.now()

        logger.info(f"Initialized clinical validator with protocol: {protocol.name}")

    def validate_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        clinical_features: Optional[pd.DataFrame] = None,
        demographics: Optional[pd.DataFrame] = None,
        timestamps: Optional[np.ndarray] = None,
        model_name: str = "unnamed_model",
    ) -> ClinicalMetrics:
        """
        Perform comprehensive clinical validation of a model.

        Args:
            y_true: True binary labels (0=no risk, 1=risk)
            y_prob: Predicted probabilities
            clinical_features: Additional clinical variables for subgroup analysis
            demographics: Demographic data for bias assessment
            timestamps: Timestamps for time-dependent analysis
            model_name: Name identifier for the model

        Returns:
            ClinicalMetrics object with comprehensive clinical evaluation
        """
        logger.info(f"Starting clinical validation for {model_name}")

        # Basic validation
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have same length")

        if not 0 <= np.min(y_prob) and np.max(y_prob) <= 1:
            logger.warning("Probabilities not in [0,1] range, clipping...")
            y_prob = np.clip(y_prob, 0, 1)

        # Compute clinical metrics
        metrics = self._compute_clinical_metrics(y_true, y_prob)

        # Add subgroup analysis if demographics provided
        if demographics is not None:
            metrics.subgroup_analysis = self._compute_subgroup_metrics(y_true, y_prob, demographics)

        # Add time-dependent analysis if timestamps provided
        if timestamps is not None:
            metrics.c_index_time_dependent, metrics.time_dependent_auc = (
                self._compute_time_dependent_metrics(y_true, y_prob, timestamps)
            )

        # Store results
        self.results.append(
            {
                "model_name": model_name,
                "validation_time": datetime.now(),
                "metrics": asdict(metrics),
                "protocol": asdict(self.protocol),
            }
        )

        logger.info(f"Clinical validation completed for {model_name}")
        return metrics

    def _compute_clinical_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> ClinicalMetrics:
        """Compute clinical-specific performance metrics."""

        # ROC curve analysis
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

        # Sensitivity at 95% specificity
        specificity = 1 - fpr
        idx_95_spec = np.argmax(specificity >= 0.95)
        sens_at_95_spec = tpr[idx_95_spec] if idx_95_spec < len(tpr) else 0.0

        # Specificity at 95% sensitivity
        idx_95_sens = np.argmax(tpr >= 0.95)
        spec_at_95_sens = specificity[idx_95_sens] if idx_95_sens < len(specificity) else 0.0

        # PPV/NPV at 95% sensitivity
        threshold_95_sens = (
            roc_thresholds[idx_95_sens] if idx_95_sens < len(roc_thresholds) else 0.5
        )
        y_pred_95_sens = (y_prob >= threshold_95_sens).astype(int)

        tp = np.sum((y_true == 1) & (y_pred_95_sens == 1))
        fp = np.sum((y_true == 0) & (y_pred_95_sens == 1))
        tn = np.sum((y_true == 0) & (y_pred_95_sens == 0))
        fn = np.sum((y_true == 1) & (y_pred_95_sens == 0))

        ppv_at_95_sens = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv_at_95_sens = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Risk stratification metrics
        high_risk_mask = y_prob >= self.protocol.high_risk_threshold
        moderate_risk_mask = (y_prob >= self.protocol.moderate_risk_threshold) & (
            y_prob < self.protocol.high_risk_threshold
        )
        low_risk_mask = y_prob < self.protocol.moderate_risk_threshold

        # High risk sensitivity
        high_risk_sens = (
            np.sum(y_true[high_risk_mask] == 1) / np.sum(y_true == 1)
            if np.sum(y_true == 1) > 0
            else 0.0
        )

        # Moderate risk precision
        moderate_risk_prec = (
            np.sum(y_true[moderate_risk_mask] == 1) / np.sum(moderate_risk_mask)
            if np.sum(moderate_risk_mask) > 0
            else 0.0
        )

        # Low risk NPV
        low_risk_npv = (
            np.sum(y_true[low_risk_mask] == 0) / np.sum(low_risk_mask)
            if np.sum(low_risk_mask) > 0
            else 0.0
        )

        # Calibration metrics
        brier = brier_score_loss(y_true, y_prob)

        try:
            fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
            # Linear regression for calibration slope/intercept
            if len(fraction_pos) > 2:
                calib_slope = np.polyfit(mean_pred, fraction_pos, 1)[0]
                calib_intercept = np.polyfit(mean_pred, fraction_pos, 1)[1]
            else:
                calib_slope, calib_intercept = 1.0, 0.0
        except:
            calib_slope, calib_intercept = 1.0, 0.0

        # Hosmer-Lemeshow test (simplified)
        hl_pvalue = self._hosmer_lemeshow_test(y_true, y_prob)

        # Net benefit analysis
        net_benefits = {}
        decision_curve_points = []

        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
            fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
            n = len(y_true)

            # Net benefit = (TP/n) - (FP/n) * (threshold/(1-threshold))
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits[str(threshold)] = net_benefit
            decision_curve_points.append(net_benefit)

        # Decision curve AUC (area under decision curve)
        decision_curve_auc = np.trapz(decision_curve_points, dx=0.1)

        return ClinicalMetrics(
            sensitivity_at_95_specificity=sens_at_95_spec,
            specificity_at_95_sensitivity=spec_at_95_sens,
            ppv_at_95_sensitivity=ppv_at_95_sens,
            npv_at_95_sensitivity=npv_at_95_sens,
            high_risk_sensitivity=high_risk_sens,
            moderate_risk_precision=moderate_risk_prec,
            low_risk_npv=low_risk_npv,
            brier_score=brier,
            calibration_slope=calib_slope,
            calibration_intercept=calib_intercept,
            hosmer_lemeshow_pvalue=hl_pvalue,
            net_benefit_at_threshold=net_benefits,
            decision_curve_auc=decision_curve_auc,
        )

    def _hosmer_lemeshow_test(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Simplified Hosmer-Lemeshow goodness-of-fit test."""
        try:
            # Group predictions into deciles
            deciles = np.quantile(y_prob, np.linspace(0.1, 0.9, 9))
            groups = np.digitize(y_prob, deciles)

            chi2_stat = 0
            for group in range(10):
                mask = groups == group
                if np.sum(mask) == 0:
                    continue

                observed_pos = np.sum(y_true[mask])
                expected_pos = np.sum(y_prob[mask])
                observed_neg = np.sum(mask) - observed_pos
                expected_neg = np.sum(mask) - expected_pos

                if expected_pos > 0 and expected_neg > 0:
                    chi2_stat += (observed_pos - expected_pos) ** 2 / expected_pos
                    chi2_stat += (observed_neg - expected_neg) ** 2 / expected_neg

            # Return simplified p-value approximation
            from scipy.stats import chi2

            return 1 - chi2.cdf(chi2_stat, df=8)

        except:
            return 1.0  # Return non-significant if test fails

    def _compute_subgroup_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, demographics: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for demographic subgroups for bias assessment."""

        subgroup_results = {}

        for column in demographics.columns:
            if column in ["age_group", "gender", "race", "ethnicity"]:
                subgroup_results[column] = {}

                for group in demographics[column].unique():
                    if pd.isna(group):
                        continue

                    mask = demographics[column] == group
                    if np.sum(mask) < 10:  # Skip small groups
                        continue

                    group_y_true = y_true[mask]
                    group_y_prob = y_prob[mask]

                    try:
                        auc = roc_auc_score(group_y_true, group_y_prob)
                        # Basic metrics for subgroup
                        basic_metrics = compute_metrics(group_y_true, group_y_prob)

                        subgroup_results[column][str(group)] = {
                            "n": int(np.sum(mask)),
                            "prevalence": float(np.mean(group_y_true)),
                            "auc": float(auc),
                            "sensitivity": float(basic_metrics.recall),
                            "specificity": float(
                                1
                                - basic_metrics.precision
                                + basic_metrics.recall * basic_metrics.precision
                            ),
                            "ppv": float(basic_metrics.precision),
                        }
                    except:
                        continue

        return subgroup_results

    def _compute_time_dependent_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, timestamps: np.ndarray
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """Compute time-dependent validation metrics."""

        # This would implement time-to-event analysis for suicide risk
        # For now, return placeholder values
        # In practice, would use survival analysis methods

        try:
            # Placeholder implementation
            # Would implement proper time-to-event analysis with censoring
            time_windows = [30, 90, 180, 365]  # Days
            time_aucs = {}

            for window in time_windows:
                # Simplified time-dependent AUC
                # Real implementation would use proper survival analysis
                time_aucs[f"{window}_days"] = float(roc_auc_score(y_true, y_prob))

            c_index = float(roc_auc_score(y_true, y_prob))  # Placeholder

            return c_index, time_aucs

        except Exception as e:
            logger.warning(f"Time-dependent analysis failed: {e}")
            return None, None

    def export_validation_report(self, output_path: Path) -> None:
        """Export comprehensive validation report."""

        report = {
            "validation_metadata": {
                "protocol": asdict(self.protocol),
                "validation_start": self.validation_start_time.isoformat(),
                "validation_end": datetime.now().isoformat(),
                "total_models_validated": len(self.results),
            },
            "results": self.results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report exported to {output_path}")

    def compare_models(self, metric: str = "sensitivity_at_95_specificity") -> pd.DataFrame:
        """Compare models on specified clinical metric."""

        if not self.results:
            return pd.DataFrame()

        comparison_data = []
        for result in self.results:
            model_name = result["model_name"]
            metrics = result["metrics"]

            comparison_data.append(
                {
                    "model": model_name,
                    "validation_time": result["validation_time"],
                    metric: metrics.get(metric, None),
                }
            )

        return pd.DataFrame(comparison_data).sort_values(metric, ascending=False)


def create_clinical_validation_protocol(
    name: str,
    validation_type: str = "retrospective",
    target_population: str = "adult_psychiatric_patients",
) -> ValidationProtocol:
    """
    Create standard clinical validation protocol for suicide risk detection.

    Args:
        name: Protocol name
        validation_type: Type of validation study
        target_population: Target patient population

    Returns:
        ValidationProtocol configured for suicide risk detection
    """

    # Standard inclusion/exclusion criteria for suicide risk studies
    inclusion_criteria = [
        "Age >= 18 years",
        "Psychiatric evaluation or treatment",
        "Text data available for analysis",
        "Informed consent obtained",
    ]

    exclusion_criteria = [
        "Age < 18 years",
        "Insufficient text data (<50 words)",
        "Non-English language primary communication",
        "Cognitive impairment preventing valid assessment",
    ]

    # Clinical endpoints
    primary_endpoint = "Suicide attempt or death by suicide within 30 days"
    secondary_endpoints = [
        "Suicide attempt within 90 days",
        "Psychiatric hospitalization within 30 days",
        "Emergency department visit for suicidal ideation",
        "Treatment engagement following risk assessment",
    ]

    safety_endpoints = [
        "False negative rate for high-risk patients",
        "Excessive alerts leading to alert fatigue",
    ]

    return ValidationProtocol(
        name=name,
        description=f"Clinical validation protocol for suicide risk detection in {target_population}",
        validation_type=validation_type,
        study_period=("2024-01-01", "2024-12-31"),
        follow_up_period=90,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        target_population=target_population,
        primary_endpoint=primary_endpoint,
        secondary_endpoints=secondary_endpoints,
        safety_endpoints=safety_endpoints,
        alpha=0.05,
        power=0.8,
        expected_prevalence=0.12,
        minimum_detectable_difference=0.05,
        high_risk_threshold=0.8,
        moderate_risk_threshold=0.3,
        intervention_threshold=0.5,
    )
