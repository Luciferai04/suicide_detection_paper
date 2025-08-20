"""
Clinical Validation and Integration Module

This module provides clinical-grade validation, risk stratification,
and healthcare system integration for suicide risk detection models.

Components:
- clinical_validation: Validation protocols and metrics
- risk_stratification: DSM-5 aligned risk scoring
- intervention_recommendations: Evidence-based intervention mapping
- healthcare_integration: FHIR/HL7 integration for EHR systems
"""

__version__ = "1.0.0"
__author__ = "Suicide Detection Research Team"
__email__ = "research@suicidedetection.org"

from .clinical_validation import ClinicalValidator, ValidationProtocol
from .healthcare_integration import FHIRConnector, HL7Gateway
from .intervention_recommendations import CarePlanGenerator, InterventionEngine
from .risk_stratification import DSM5Classifier, RiskStratifier

__all__ = [
    "ClinicalValidator",
    "ValidationProtocol",
    "RiskStratifier",
    "DSM5Classifier",
    "InterventionEngine",
    "CarePlanGenerator",
    "FHIRConnector",
    "HL7Gateway",
]
