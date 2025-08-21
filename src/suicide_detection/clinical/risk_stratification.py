"""
Risk Stratification and DSM-5 Integration

Provides DSM-5 aligned risk stratification, severity classification,
and clinical decision support for suicide risk assessment.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """DSM-5 aligned suicide risk levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    IMMINENT = "imminent"


class SeverityLevel(Enum):
    """DSM-5 severity classifications."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    SEVERE_WITH_PSYCHOTIC_FEATURES = "severe_with_psychotic_features"


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result."""

    # Core risk metrics
    risk_level: RiskLevel
    risk_probability: float
    severity_level: SeverityLevel

    # Clinical indicators
    suicidal_ideation_severity: int  # 0-10 scale
    intent_severity: int  # 0-10 scale
    capability_score: int  # 0-10 scale
    hopelessness_score: int  # 0-10 scale

    # Risk factors
    acute_risk_factors: List[str]
    chronic_risk_factors: List[str]
    protective_factors: List[str]

    # Clinical context
    current_stressors: List[str]
    psychiatric_comorbidities: List[str]
    substance_use_indicators: List[str]

    # Time-based assessment
    assessment_timestamp: datetime
    reassessment_recommended: datetime
    crisis_contact_indicated: bool

    # Confidence and reliability
    prediction_confidence: float
    model_version: str
    clinical_override_notes: Optional[str] = None


@dataclass
class DSM5Criteria:
    """DSM-5 criteria for suicide risk assessment."""

    # Major Depressive Episode criteria
    depressed_mood: bool = False
    anhedonia: bool = False
    appetite_change: bool = False
    sleep_disturbance: bool = False
    psychomotor_changes: bool = False
    fatigue: bool = False
    worthlessness_guilt: bool = False
    concentration_problems: bool = False
    death_thoughts: bool = False

    # Suicidal ideation specific
    passive_ideation: bool = False
    active_ideation: bool = False
    intent_to_act: bool = False
    specific_plan: bool = False
    access_to_means: bool = False

    # Additional risk factors
    previous_attempts: int = 0
    family_history_suicide: bool = False
    substance_use_disorder: bool = False
    personality_disorder: bool = False
    psychotic_symptoms: bool = False

    # Protective factors
    social_support: bool = False
    treatment_engagement: bool = False
    religious_beliefs: bool = False
    responsibility_to_others: bool = False


class RiskStratifier:
    """
    Risk stratification engine using DSM-5 aligned criteria
    and evidence-based risk assessment frameworks.
    """

    def __init__(self, model_version: str = "1.0.0"):
        """
        Initialize risk stratification engine.

        Args:
            model_version: Version identifier for risk model
        """
        self.model_version = model_version
        self.risk_thresholds = {
            RiskLevel.LOW: (0.0, 0.3),
            RiskLevel.MODERATE: (0.3, 0.7),
            RiskLevel.HIGH: (0.7, 0.9),
            RiskLevel.IMMINENT: (0.9, 1.0),
        }

        # Evidence-based risk factor weights
        self.risk_factor_weights = {
            # Acute risk factors (higher weights)
            "active_suicidal_ideation": 0.25,
            "suicide_plan": 0.30,
            "recent_attempt": 0.35,
            "access_to_means": 0.20,
            "hopelessness": 0.15,
            "agitation": 0.12,
            "psychosis": 0.18,
            "substance_intoxication": 0.16,
            # Chronic risk factors (moderate weights)
            "depression_severity": 0.10,
            "previous_attempts": 0.12,
            "family_history": 0.08,
            "substance_use_disorder": 0.09,
            "personality_disorder": 0.07,
            "social_isolation": 0.08,
            "chronic_pain": 0.06,
            # Protective factors (negative weights)
            "social_support": -0.10,
            "treatment_engagement": -0.08,
            "religious_beliefs": -0.06,
            "responsibility_others": -0.09,
            "future_orientation": -0.07,
        }

        logger.info(f"Initialized risk stratifier version {model_version}")

    def assess_risk(
        self,
        ml_prediction: float,
        clinical_features: Dict[str, Any],
        dsm5_criteria: Optional[DSM5Criteria] = None,
        patient_history: Optional[Dict[str, Any]] = None,
        current_context: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment combining ML prediction with clinical factors.

        Args:
            ml_prediction: Model probability output (0-1)
            clinical_features: Clinical feature dictionary
            dsm5_criteria: DSM-5 criteria assessment
            patient_history: Historical patient information
            current_context: Current clinical context

        Returns:
            RiskAssessment with comprehensive evaluation
        """

        # Combine ML prediction with clinical risk factors
        adjusted_risk = self._compute_adjusted_risk(ml_prediction, clinical_features, dsm5_criteria)

        # Determine risk level
        risk_level = self._classify_risk_level(adjusted_risk)

        # Assess severity using DSM-5 criteria
        severity_level = self._assess_severity(dsm5_criteria, clinical_features)

        # Extract clinical indicators
        indicators = self._extract_clinical_indicators(clinical_features, dsm5_criteria)

        # Identify risk and protective factors
        acute_factors, chronic_factors, protective_factors = self._identify_risk_factors(
            clinical_features, patient_history
        )

        # Assess current context
        stressors, comorbidities, substance_indicators = self._assess_current_context(
            current_context or {}
        )

        # Determine timing recommendations
        reassessment_time, crisis_contact = self._determine_clinical_actions(
            risk_level, severity_level
        )

        # Calculate confidence
        confidence = self._calculate_prediction_confidence(ml_prediction, clinical_features)

        return RiskAssessment(
            risk_level=risk_level,
            risk_probability=adjusted_risk,
            severity_level=severity_level,
            suicidal_ideation_severity=indicators["ideation_severity"],
            intent_severity=indicators["intent_severity"],
            capability_score=indicators["capability_score"],
            hopelessness_score=indicators["hopelessness_score"],
            acute_risk_factors=acute_factors,
            chronic_risk_factors=chronic_factors,
            protective_factors=protective_factors,
            current_stressors=stressors,
            psychiatric_comorbidities=comorbidities,
            substance_use_indicators=substance_indicators,
            assessment_timestamp=datetime.now(),
            reassessment_recommended=reassessment_time,
            crisis_contact_indicated=crisis_contact,
            prediction_confidence=confidence,
            model_version=self.model_version,
        )

    def _compute_adjusted_risk(
        self,
        ml_prediction: float,
        clinical_features: Dict[str, Any],
        dsm5_criteria: Optional[DSM5Criteria],
    ) -> float:
        """Adjust ML prediction using clinical risk factors."""

        base_risk = ml_prediction
        risk_adjustment = 0.0

        # Add adjustments from clinical features
        risk_adjustment += self._adjust_for_features(clinical_features)

        # Add adjustments from DSM-5 criteria
        if dsm5_criteria is not None:
            risk_adjustment += self._adjust_for_dsm5(dsm5_criteria)

        # Combine and bound
        adjusted_risk = base_risk + risk_adjustment
        return max(0.0, min(1.0, adjusted_risk))

    def _adjust_for_features(self, clinical_features: Dict[str, Any]) -> float:
        """Compute risk adjustment from clinical feature dictionary."""
        adj = 0.0
        for factor, weight in self.risk_factor_weights.items():
            if factor not in clinical_features:
                continue
            feature_value = clinical_features[factor]
            if isinstance(feature_value, bool):
                adjustment = weight if feature_value else 0.0
            elif isinstance(feature_value, (int, float)):
                normalized_value = min(feature_value / 10.0, 1.0)
                adjustment = weight * normalized_value
            else:
                adjustment = 0.0
            adj += adjustment
        return adj

    def _adjust_for_dsm5(self, dsm5_criteria: DSM5Criteria) -> float:
        """Compute risk adjustment from DSM-5 criteria block."""
        adj = 0.0
        # Major suicide-specific criteria
        if dsm5_criteria.active_ideation:
            adj += 0.20
        if dsm5_criteria.specific_plan:
            adj += 0.25
        if dsm5_criteria.access_to_means:
            adj += 0.15
        if dsm5_criteria.intent_to_act:
            adj += 0.30
        # Previous attempts (escalating weight)
        if dsm5_criteria.previous_attempts > 0:
            adj += min(dsm5_criteria.previous_attempts * 0.08, 0.25)
        # Protective factors diminish risk
        protective_count = sum(
            [
                dsm5_criteria.social_support,
                dsm5_criteria.treatment_engagement,
                dsm5_criteria.religious_beliefs,
                dsm5_criteria.responsibility_to_others,
            ]
        )
        adj -= protective_count * 0.05
        return adj

    def _classify_risk_level(self, risk_probability: float) -> RiskLevel:
        """Classify risk level based on probability."""

        for level, (min_thresh, max_thresh) in self.risk_thresholds.items():
            if min_thresh <= risk_probability < max_thresh:
                return level

        return RiskLevel.IMMINENT  # Default to highest if >= 1.0

    def _assess_severity(
        self, dsm5_criteria: Optional[DSM5Criteria], clinical_features: Dict[str, Any]
    ) -> SeverityLevel:
        """Assess DSM-5 aligned severity level."""

        if dsm5_criteria is None:
            return SeverityLevel.MODERATE  # Default if no criteria

        # Count major depressive episode criteria
        mde_criteria_count = sum(
            [
                dsm5_criteria.depressed_mood,
                dsm5_criteria.anhedonia,
                dsm5_criteria.appetite_change,
                dsm5_criteria.sleep_disturbance,
                dsm5_criteria.psychomotor_changes,
                dsm5_criteria.fatigue,
                dsm5_criteria.worthlessness_guilt,
                dsm5_criteria.concentration_problems,
                dsm5_criteria.death_thoughts,
            ]
        )

        # Assess functional impairment (if available)
        functional_impairment = clinical_features.get("functional_impairment", 0)

        # Determine severity
        if dsm5_criteria.psychotic_symptoms:
            return SeverityLevel.SEVERE_WITH_PSYCHOTIC_FEATURES
        elif mde_criteria_count >= 7 or functional_impairment >= 8:
            return SeverityLevel.SEVERE
        elif mde_criteria_count >= 5 or functional_impairment >= 5:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.MILD

    def _extract_clinical_indicators(
        self, clinical_features: Dict[str, Any], dsm5_criteria: Optional[DSM5Criteria]
    ) -> Dict[str, int]:
        """Extract clinical indicator scores (0-10 scale)."""

        indicators = {
            "ideation_severity": 0,
            "intent_severity": 0,
            "capability_score": 0,
            "hopelessness_score": 0,
        }

        # Suicidal ideation severity
        if dsm5_criteria:
            if dsm5_criteria.passive_ideation:
                indicators["ideation_severity"] = 3
            if dsm5_criteria.active_ideation:
                indicators["ideation_severity"] = 6
            if dsm5_criteria.specific_plan:
                indicators["ideation_severity"] = 8
            if dsm5_criteria.intent_to_act:
                indicators["intent_severity"] = 8

        # Use clinical features if available
        for indicator in indicators.keys():
            if indicator in clinical_features:
                indicators[indicator] = min(int(clinical_features[indicator]), 10)

        # Capability assessment
        capability_factors = [
            clinical_features.get("access_to_means", False),
            clinical_features.get("previous_attempts", 0) > 0,
            clinical_features.get("acquired_capability", False),
            clinical_features.get("fearlessness_about_death", False),
        ]
        indicators["capability_score"] = sum(capability_factors) * 2.5

        # Hopelessness (Beck Hopelessness Scale proxy)
        hopelessness_items = [
            clinical_features.get("future_negative", False),
            clinical_features.get("no_motivation", False),
            clinical_features.get("expect_failure", False),
            clinical_features.get("no_satisfaction_anticipated", False),
        ]
        indicators["hopelessness_score"] = sum(hopelessness_items) * 2.5

        return indicators

    def _identify_risk_factors(
        self, clinical_features: Dict[str, Any], patient_history: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Identify acute, chronic, and protective factors."""

        acute_factors = []
        chronic_factors = []
        protective_factors = []

        # Acute risk factors
        acute_factor_mapping = {
            "recent_stressor": "Recent major life stressor",
            "relationship_loss": "Recent relationship loss",
            "job_loss": "Recent employment loss",
            "financial_crisis": "Financial crisis",
            "legal_problems": "Legal problems",
            "substance_intoxication": "Current substance intoxication",
            "psychotic_episode": "Active psychotic symptoms",
            "severe_anxiety": "Severe anxiety/panic",
            "insomnia": "Severe insomnia",
            "agitation": "Psychomotor agitation",
        }

        for key, description in acute_factor_mapping.items():
            if clinical_features.get(key, False):
                acute_factors.append(description)

        # Chronic risk factors
        chronic_factor_mapping = {
            "depression_history": "History of major depression",
            "bipolar_disorder": "Bipolar disorder",
            "substance_use_disorder": "Substance use disorder",
            "personality_disorder": "Personality disorder",
            "trauma_history": "History of trauma/PTSD",
            "chronic_pain": "Chronic pain condition",
            "medical_illness": "Serious medical illness",
            "social_isolation": "Social isolation",
            "unemployment": "Chronic unemployment",
        }

        for key, description in chronic_factor_mapping.items():
            if clinical_features.get(key, False):
                chronic_factors.append(description)

        # Add patient history factors
        if patient_history:
            if patient_history.get("previous_attempts", 0) > 0:
                chronic_factors.append(
                    f"Previous suicide attempts ({patient_history['previous_attempts']})"
                )
            if patient_history.get("family_history_suicide", False):
                chronic_factors.append("Family history of suicide")

        # Protective factors
        protective_factor_mapping = {
            "social_support": "Strong social support system",
            "treatment_engagement": "Active in mental health treatment",
            "religious_beliefs": "Religious/spiritual beliefs",
            "responsibility_others": "Responsibility for dependents",
            "future_plans": "Future-oriented goals/plans",
            "coping_skills": "Effective coping strategies",
            "medication_adherence": "Good medication compliance",
            "employment": "Stable employment",
        }

        for key, description in protective_factor_mapping.items():
            if clinical_features.get(key, False):
                protective_factors.append(description)

        return acute_factors, chronic_factors, protective_factors

    def _assess_current_context(
        self, current_context: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Assess current clinical context."""

        stressors = current_context.get("current_stressors", [])
        if isinstance(stressors, str):
            stressors = [stressors]

        comorbidities = current_context.get("psychiatric_comorbidities", [])
        if isinstance(comorbidities, str):
            comorbidities = [comorbidities]

        substance_indicators = current_context.get("substance_use_indicators", [])
        if isinstance(substance_indicators, str):
            substance_indicators = [substance_indicators]

        return stressors, comorbidities, substance_indicators

    def _determine_clinical_actions(
        self, risk_level: RiskLevel, severity_level: SeverityLevel
    ) -> Tuple[datetime, bool]:
        """Determine reassessment timing and crisis contact need."""

        now = datetime.now()

        # Reassessment timing based on risk level
        if risk_level == RiskLevel.IMMINENT:
            reassessment = now + timedelta(hours=1)
            crisis_contact = True
        elif risk_level == RiskLevel.HIGH:
            reassessment = now + timedelta(hours=4)
            crisis_contact = True
        elif risk_level == RiskLevel.MODERATE:
            reassessment = now + timedelta(hours=24)
            crisis_contact = False
        else:  # LOW risk
            reassessment = now + timedelta(days=7)
            crisis_contact = False

        # Adjust based on severity
        if severity_level == SeverityLevel.SEVERE_WITH_PSYCHOTIC_FEATURES:
            reassessment = min(reassessment, now + timedelta(hours=2))
            crisis_contact = True

        return reassessment, crisis_contact

    def _calculate_prediction_confidence(
        self, ml_prediction: float, clinical_features: Dict[str, Any]
    ) -> float:
        """Calculate prediction confidence based on available information."""

        base_confidence = 0.5

        # Increase confidence with more clinical information
        feature_completeness = len(clinical_features) / 20.0  # Assume 20 key features
        base_confidence += feature_completeness * 0.3

        # Increase confidence if prediction is not near decision boundaries
        boundary_distance = min(
            abs(ml_prediction - 0.3), abs(ml_prediction - 0.7), abs(ml_prediction - 0.9)
        )
        base_confidence += boundary_distance * 0.2

        return min(base_confidence, 1.0)


class DSM5Classifier:
    """
    DSM-5 criteria classifier for automated clinical assessment.
    """

    def __init__(self):
        """Initialize DSM-5 classifier."""

        # Text patterns for criteria identification
        self.criteria_patterns = {
            # Major Depressive Episode
            "depressed_mood": [
                "depressed",
                "sad",
                "empty",
                "hopeless",
                "down mood",
                "feeling low",
                "melancholy",
                "blue",
            ],
            "anhedonia": [
                "no interest",
                "no pleasure",
                "anhedonia",
                "lost enjoyment",
                "nothing fun",
                "no motivation",
            ],
            "appetite_change": [
                "appetite loss",
                "not eating",
                "overeating",
                "weight loss",
                "weight gain",
                "food problems",
            ],
            # Suicidal ideation
            "passive_ideation": [
                "wish I was dead",
                "better off dead",
                "want to die",
                "tired of living",
                "no point living",
            ],
            "active_ideation": [
                "kill myself",
                "suicide",
                "end my life",
                "take my life",
                "hurt myself",
                "self harm",
            ],
            "specific_plan": [
                "plan to",
                "how I would",
                "method",
                "pills",
                "gun",
                "rope",
                "jump",
                "overdose",
            ],
        }

        logger.info("Initialized DSM-5 classifier")

    def classify_text(
        self, text: str, additional_features: Optional[Dict[str, Any]] = None
    ) -> DSM5Criteria:
        """
        Classify text against DSM-5 criteria.

        Args:
            text: Input text to analyze
            additional_features: Additional structured clinical information

        Returns:
            DSM5Criteria object with binary classifications
        """

        text_lower = text.lower()
        criteria = DSM5Criteria()

        # Pattern matching for text-based criteria
        for criterion, patterns in self.criteria_patterns.items():
            if hasattr(criteria, criterion):
                criterion_present = any(pattern in text_lower for pattern in patterns)
                setattr(criteria, criterion, criterion_present)

        # Incorporate additional structured features if provided
        if additional_features:
            for key, value in additional_features.items():
                if hasattr(criteria, key) and isinstance(value, bool):
                    setattr(criteria, key, value)
                elif hasattr(criteria, key) and isinstance(value, (int, float)):
                    setattr(criteria, key, int(value))

        return criteria

    def score_criteria_completeness(self, criteria: DSM5Criteria) -> Dict[str, float]:
        """Score completeness of criteria assessment."""

        # Major Depressive Episode criteria
        mde_criteria = [
            criteria.depressed_mood,
            criteria.anhedonia,
            criteria.appetite_change,
            criteria.sleep_disturbance,
            criteria.psychomotor_changes,
            criteria.fatigue,
            criteria.worthlessness_guilt,
            criteria.concentration_problems,
            criteria.death_thoughts,
        ]

        # Suicidal ideation criteria
        si_criteria = [
            criteria.passive_ideation,
            criteria.active_ideation,
            criteria.intent_to_act,
            criteria.specific_plan,
            criteria.access_to_means,
        ]

        # Additional risk factors
        risk_factors = [
            criteria.previous_attempts > 0,
            criteria.family_history_suicide,
            criteria.substance_use_disorder,
            criteria.personality_disorder,
            criteria.psychotic_symptoms,
        ]

        return {
            "mde_completeness": sum(mde_criteria) / len(mde_criteria),
            "suicidal_ideation_completeness": sum(si_criteria) / len(si_criteria),
            "risk_factors_completeness": sum(risk_factors) / len(risk_factors),
            "overall_completeness": sum(mde_criteria + si_criteria + risk_factors)
            / (len(mde_criteria) + len(si_criteria) + len(risk_factors)),
        }
