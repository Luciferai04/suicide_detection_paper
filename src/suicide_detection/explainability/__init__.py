"""
Explainability and Interpretability Framework

Provides advanced interpretability capabilities for suicide risk detection models
including SHAP analysis, attention visualization, and clinical explanations.
"""

from .attention_visualizer import AttentionMap, AttentionVisualizer
from .clinical_explainer import ClinicalExplainer, ClinicalExplanation
from .feature_importance import FeatureImportanceAnalyzer, ImportanceResults
from .shap_explainer import SHAPExplainer, SHAPResults

__all__ = [
    "SHAPExplainer",
    "SHAPResults",
    "AttentionVisualizer",
    "AttentionMap",
    "ClinicalExplainer",
    "ClinicalExplanation",
    "FeatureImportanceAnalyzer",
    "ImportanceResults",
]
