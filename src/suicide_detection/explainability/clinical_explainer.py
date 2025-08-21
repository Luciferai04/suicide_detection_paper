"""
Clinical Explainability Framework

Provides clinician-friendly explanations of model predictions with
domain-specific insights and actionable recommendations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClinicalExplanation:
    """Clinical explanation of model prediction."""

    # Prediction details
    prediction_probability: float
    risk_level: str
    confidence_score: float

    # Top contributing factors
    high_risk_indicators: List[Dict[str, Any]]
    protective_factors: List[Dict[str, Any]]

    # Clinical interpretation
    primary_concerns: List[str]
    clinical_recommendations: List[str]

    # Evidence and context
    similar_cases: Optional[Dict[str, Any]] = None
    literature_references: List[str] = None

    # Uncertainty and limitations
    prediction_uncertainty: float = 0.0
    model_limitations: List[str] = None

    # Temporal factors
    explanation_timestamp: datetime = None

    def __post_init__(self):
        if self.explanation_timestamp is None:
            self.explanation_timestamp = datetime.now()
        if self.literature_references is None:
            self.literature_references = []
        if self.model_limitations is None:
            self.model_limitations = []


class ClinicalExplainer:
    """
    Clinical explainability framework for suicide risk detection models.

    Translates complex ML model outputs into clinically meaningful explanations
    that support clinical decision-making and build clinician trust.
    """

    def __init__(self):
        """Initialize clinical explainer with domain knowledge."""

        # Clinical risk factor mappings
        self.clinical_risk_factors = self._initialize_risk_factors()
        self.protective_factors = self._initialize_protective_factors()

        # Clinical interpretation templates
        self.explanation_templates = self._initialize_templates()

        # Literature references
        self.literature_base = self._initialize_literature()

        logger.info("Initialized clinical explainer framework")

    def explain_prediction(
        self,
        text: str,
        prediction: float,
        feature_importance: Optional[Dict[str, float]] = None,
        attention_weights: Optional[np.ndarray] = None,
        model_type: str = "bert",
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> ClinicalExplanation:
        """
        Generate clinical explanation for model prediction.

        Args:
            text: Input text that was analyzed
            prediction: Model prediction probability
            feature_importance: Feature importance scores (if available)
            attention_weights: Attention weights (for transformer models)
            model_type: Type of model used ("svm", "bilstm", "bert")
            patient_context: Additional patient context

        Returns:
            ClinicalExplanation with detailed clinical interpretation
        """

        # Determine risk level
        risk_level = self._classify_risk_level(prediction)

        # Extract clinical indicators from text
        text_indicators = self._extract_clinical_indicators(text)

        # Analyze feature importance or attention
        if feature_importance is not None:
            important_features = self._analyze_feature_importance(feature_importance)
        elif attention_weights is not None:
            important_features = self._analyze_attention_patterns(text, attention_weights)
        else:
            important_features = {"high_risk": [], "protective": []}

        # Combine text indicators with model features
        high_risk_indicators = self._combine_risk_indicators(
            text_indicators["risk_factors"], important_features["high_risk"], patient_context or {}
        )

        protective_factors = self._combine_protective_factors(
            text_indicators["protective_factors"],
            important_features["protective"],
            patient_context or {},
        )

        # Generate clinical interpretation
        primary_concerns = self._identify_primary_concerns(high_risk_indicators, risk_level)
        clinical_recommendations = self._generate_recommendations(
            risk_level, high_risk_indicators, protective_factors, model_type
        )

        # Calculate confidence and uncertainty
        confidence_score = self._calculate_confidence(prediction, high_risk_indicators)
        uncertainty = self._calculate_uncertainty(prediction, model_type)

        # Find similar cases (if available)
        similar_cases = self._find_similar_cases(text_indicators, prediction)

        # Get relevant literature
        literature_refs = self._get_relevant_literature(high_risk_indicators)

        # Identify model limitations
        limitations = self._identify_limitations(model_type, text, patient_context)

        return ClinicalExplanation(
            prediction_probability=prediction,
            risk_level=risk_level,
            confidence_score=confidence_score,
            high_risk_indicators=high_risk_indicators,
            protective_factors=protective_factors,
            primary_concerns=primary_concerns,
            clinical_recommendations=clinical_recommendations,
            similar_cases=similar_cases,
            literature_references=literature_refs,
            prediction_uncertainty=uncertainty,
            model_limitations=limitations,
        )

    def generate_clinician_report(
        self, explanation: ClinicalExplanation, patient_id: str = None
    ) -> str:
        """Generate human-readable report for clinicians."""

        lines = []
        self._append_header(lines, patient_id, explanation)
        self._append_summary(lines, explanation)
        self._append_primary_concerns(lines, explanation)
        self._append_high_risk_indicators(lines, explanation)
        self._append_protective_factors(lines, explanation)
        self._append_recommendations(lines, explanation)
        self._append_limitations(lines, explanation)
        self._append_literature(lines, explanation)
        self._append_disclaimer(lines)
        return "\n".join(lines)

    def _append_header(self, lines: list[str], patient_id: str | None, explanation: ClinicalExplanation) -> None:
        lines.append("=" * 60)
        lines.append("SUICIDE RISK ASSESSMENT - CLINICAL EXPLANATION")
        lines.append("=" * 60)
        if patient_id:
            lines.append(f"Patient ID: {patient_id}")
        lines.append(
            f"Assessment Time: {explanation.explanation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("")

    def _append_summary(self, lines: list[str], explanation: ClinicalExplanation) -> None:
        lines.append("RISK ASSESSMENT SUMMARY")
        lines.append("-" * 30)
        lines.append(f"Risk Level: {explanation.risk_level.upper()}")
        lines.append(f"Prediction Probability: {explanation.prediction_probability:.3f}")
        lines.append(f"Confidence Score: {explanation.confidence_score:.3f}")
        if explanation.prediction_uncertainty > 0.2:
            lines.append(f"⚠️  High Uncertainty: {explanation.prediction_uncertainty:.3f}")
        lines.append("")

    def _append_primary_concerns(self, lines: list[str], explanation: ClinicalExplanation) -> None:
        if not explanation.primary_concerns:
            return
        lines.append("PRIMARY CLINICAL CONCERNS")
        lines.append("-" * 30)
        for concern in explanation.primary_concerns:
            lines.append(f"• {concern}")
        lines.append("")

    def _append_high_risk_indicators(self, lines: list[str], explanation: ClinicalExplanation) -> None:
        if not explanation.high_risk_indicators:
            return
        lines.append("HIGH-RISK INDICATORS IDENTIFIED")
        lines.append("-" * 30)
        for indicator in explanation.high_risk_indicators[:5]:
            importance = indicator.get("importance", 0)
            evidence = indicator.get("evidence", "N/A")
            lines.append(f"• {indicator['factor']} (importance: {importance:.3f})")
            if evidence != "N/A":
                lines.append(f"  Evidence: {evidence}")
        lines.append("")

    def _append_protective_factors(self, lines: list[str], explanation: ClinicalExplanation) -> None:
        if not explanation.protective_factors:
            return
        lines.append("PROTECTIVE FACTORS PRESENT")
        lines.append("-" * 30)
        for factor in explanation.protective_factors[:3]:
            lines.append(f"• {factor['factor']} (strength: {factor.get('strength', 0):.3f})")
        lines.append("")

    def _append_recommendations(self, lines: list[str], explanation: ClinicalExplanation) -> None:
        lines.append("CLINICAL RECOMMENDATIONS")
        lines.append("-" * 30)
        for i, rec in enumerate(explanation.clinical_recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    def _append_limitations(self, lines: list[str], explanation: ClinicalExplanation) -> None:
        if not explanation.model_limitations:
            return
        lines.append("MODEL LIMITATIONS & CONSIDERATIONS")
        lines.append("-" * 30)
        for limitation in explanation.model_limitations:
            lines.append(f"• {limitation}")
        lines.append("")

    def _append_literature(self, lines: list[str], explanation: ClinicalExplanation) -> None:
        if not explanation.literature_references:
            return
        lines.append("SUPPORTING LITERATURE")
        lines.append("-" * 30)
        for ref in explanation.literature_references[:3]:
            lines.append(f"• {ref}")
        lines.append("")

    def _append_disclaimer(self, lines: list[str]) -> None:
        lines.append("IMPORTANT CLINICAL DISCLAIMER")
        lines.append("-" * 30)
        lines.append("This AI-generated assessment is intended to support, not replace,")
        lines.append("clinical judgment. Always consider the full clinical context and")
        lines.append(
            "conduct comprehensive clinical evaluation before making treatment decisions."
        )

    def _initialize_risk_factors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical risk factor mappings."""

        return {
            # Direct suicide-related expressions
            "direct_suicidal_ideation": {
                "patterns": [
                    "kill myself",
                    "end my life",
                    "suicide",
                    "take my life",
                    "hurt myself",
                ],
                "weight": 0.9,
                "clinical_significance": "Direct expression of suicidal ideation",
                "intervention": "Immediate safety assessment required",
            },
            # Hopelessness and despair
            "hopelessness": {
                "patterns": [
                    "hopeless",
                    "no way out",
                    "trapped",
                    "pointless",
                    "no future",
                    "giving up",
                ],
                "weight": 0.8,
                "clinical_significance": "Hopelessness is a strong predictor of suicide risk",
                "intervention": "Address cognitive distortions, instill hope",
            },
            # Social isolation
            "social_isolation": {
                "patterns": [
                    "alone",
                    "lonely",
                    "no one cares",
                    "isolated",
                    "no friends",
                    "abandoned",
                ],
                "weight": 0.7,
                "clinical_significance": "Social isolation increases suicide risk",
                "intervention": "Strengthen social connections and support",
            },
            # Substance use indicators
            "substance_use": {
                "patterns": ["drinking", "drugs", "high", "drunk", "stoned", "wasted", "overdose"],
                "weight": 0.6,
                "clinical_significance": "Substance use impairs judgment and increases impulsivity",
                "intervention": "Assess substance use patterns, consider treatment",
            },
            # Sleep disturbance
            "sleep_problems": {
                "patterns": [
                    "can't sleep",
                    "insomnia",
                    "nightmares",
                    "tired",
                    "exhausted",
                    "no sleep",
                ],
                "weight": 0.5,
                "clinical_significance": "Sleep disturbance common in depression and suicide risk",
                "intervention": "Address sleep hygiene, consider sleep study",
            },
            # Relationship problems
            "relationship_distress": {
                "patterns": [
                    "breakup",
                    "divorce",
                    "fighting",
                    "relationship problems",
                    "lost love",
                ],
                "weight": 0.6,
                "clinical_significance": "Relationship loss is a significant stressor",
                "intervention": "Support processing of loss, prevent isolation",
            },
            # Financial stress
            "financial_stress": {
                "patterns": ["broke", "bankrupt", "lost job", "money problems", "can't afford"],
                "weight": 0.5,
                "clinical_significance": "Financial stress can precipitate suicidal crises",
                "intervention": "Connect with financial counseling resources",
            },
        }

    def _initialize_protective_factors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize protective factor mappings."""

        return {
            "social_support": {
                "patterns": [
                    "family",
                    "friends",
                    "support",
                    "love",
                    "care about me",
                    "there for me",
                ],
                "weight": -0.6,  # Negative weight = protective
                "clinical_significance": "Social support is a key protective factor",
                "clinical_action": "Strengthen and activate support network",
            },
            "future_orientation": {
                "patterns": [
                    "tomorrow",
                    "next week",
                    "future",
                    "plans",
                    "goals",
                    "hope",
                    "looking forward",
                ],
                "weight": -0.5,
                "clinical_significance": "Future orientation reduces immediate risk",
                "clinical_action": "Reinforce future goals and planning",
            },
            "help_seeking": {
                "patterns": ["therapy", "counseling", "help", "treatment", "doctor", "medication"],
                "weight": -0.7,
                "clinical_significance": "Help-seeking behavior indicates engagement",
                "clinical_action": "Support continued treatment engagement",
            },
            "coping_strategies": {
                "patterns": ["exercise", "meditation", "music", "writing", "art", "breathing"],
                "weight": -0.4,
                "clinical_significance": "Active coping strategies reduce risk",
                "clinical_action": "Reinforce effective coping mechanisms",
            },
            "responsibility_others": {
                "patterns": ["children", "kids", "pets", "family depends", "responsible for"],
                "weight": -0.8,
                "clinical_significance": "Responsibility for others is highly protective",
                "clinical_action": "Highlight responsibilities and connections",
            },
        }

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize explanation templates."""

        return {
            "high_risk_summary": "The model identified {risk_level} suicide risk based on {num_indicators} key indicators including {top_indicators}.",
            "moderate_risk_summary": "The assessment suggests moderate risk with {num_indicators} concerning factors, particularly {top_indicators}.",
            "protective_summary": "However, {num_protective} protective factors were identified, including {top_protective}, which may help reduce risk.",
            "uncertainty_note": "This prediction has {uncertainty_level} uncertainty. Additional clinical information may improve accuracy.",
            "immediate_action": "Given the {risk_level} risk level, immediate intervention is recommended including {interventions}.",
        }

    def _initialize_literature(self) -> List[str]:
        """Initialize literature reference database."""

        return [
            "Franklin JC, et al. Risk factors for suicidal thoughts and behaviors: A meta-analysis of 50 years of research. Psychol Bull. 2017;143(2):187-232.",
            "Ribeiro JD, et al. Self-injurious thoughts and behaviors as risk factors for future suicide ideation, attempts, and death. Psychol Med. 2016;46(2):225-36.",
            "Klonsky ED, May AM, Saffer BY. Suicide, suicide attempts, and suicidal ideation. Annu Rev Clin Psychol. 2016;12:307-30.",
            "Belsher BE, et al. Prediction models for suicide attempts and deaths: a systematic review and simulation. JAMA Psychiatry. 2019;76(6):642-651.",
            "Stanley B, Brown GK. Safety planning intervention: a brief intervention to mitigate suicide risk. Cogn Behav Pract. 2012;19(2):256-264.",
            "Joiner T. Why People Die by Suicide. Harvard University Press; 2005.",
            "Van Orden KA, et al. The interpersonal theory of suicide. Psychol Rev. 2010;117(2):575-600.",
        ]

    def _classify_risk_level(self, prediction: float) -> str:
        """Classify risk level from prediction probability."""

        if prediction >= 0.9:
            return "imminent"
        elif prediction >= 0.7:
            return "high"
        elif prediction >= 0.3:
            return "moderate"
        else:
            return "low"

    def _extract_clinical_indicators(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract clinical indicators from text."""

        text_lower = text.lower()

        risk_factors = []
        protective_factors = []

        # Check for risk factors
        for factor_name, factor_info in self.clinical_risk_factors.items():
            matches = [pattern for pattern in factor_info["patterns"] if pattern in text_lower]
            if matches:
                risk_factors.append(
                    {
                        "factor": factor_name,
                        "evidence": matches,
                        "weight": factor_info["weight"],
                        "clinical_significance": factor_info["clinical_significance"],
                    }
                )

        # Check for protective factors
        for factor_name, factor_info in self.protective_factors.items():
            matches = [pattern for pattern in factor_info["patterns"] if pattern in text_lower]
            if matches:
                protective_factors.append(
                    {
                        "factor": factor_name,
                        "evidence": matches,
                        "weight": abs(factor_info["weight"]),  # Make positive for display
                        "clinical_significance": factor_info["clinical_significance"],
                    }
                )

        return {"risk_factors": risk_factors, "protective_factors": protective_factors}

    def _analyze_feature_importance(self, feature_importance: Dict[str, float]) -> Dict[str, List]:
        """Analyze feature importance scores."""

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

        high_risk = []
        protective = []

        for feature, importance in sorted_features[:10]:  # Top 10 features
            if importance > 0:
                high_risk.append(
                    {
                        "factor": feature,
                        "importance": importance,
                        "evidence": f"Feature importance: {importance:.3f}",
                    }
                )
            else:
                protective.append(
                    {
                        "factor": feature,
                        "strength": abs(importance),
                        "evidence": f"Protective importance: {abs(importance):.3f}",
                    }
                )

        return {"high_risk": high_risk, "protective": protective}

    def _analyze_attention_patterns(
        self, text: str, attention_weights: np.ndarray
    ) -> Dict[str, List]:
        """Analyze attention patterns in transformer models."""

        # This is a simplified version - in practice would need tokenizer alignment
        tokens = text.split()

        if len(attention_weights) != len(tokens):
            # Handle tokenizer mismatch
            attention_weights = attention_weights[: len(tokens)]

        # Get top attended tokens
        top_indices = np.argsort(attention_weights)[-10:]  # Top 10

        high_risk = []
        for idx in top_indices:
            if idx < len(tokens):
                token = tokens[idx]
                attention_score = attention_weights[idx]

                # Check if token relates to known risk factors
                is_risk_token = any(
                    token.lower() in pattern
                    for factor_info in self.clinical_risk_factors.values()
                    for pattern in factor_info["patterns"]
                )

                if is_risk_token:
                    high_risk.append(
                        {
                            "factor": f"Attention to: {token}",
                            "importance": float(attention_score),
                            "evidence": f"High attention weight: {attention_score:.3f}",
                        }
                    )

        return {"high_risk": high_risk, "protective": []}

    def _combine_risk_indicators(
        self, text_indicators: List[Dict], model_indicators: List[Dict], patient_context: Dict
    ) -> List[Dict[str, Any]]:
        """Combine risk indicators from different sources."""

        combined = []

        # Add text-based indicators
        for indicator in text_indicators:
            combined.append(
                {
                    "factor": indicator["factor"].replace("_", " ").title(),
                    "importance": indicator["weight"],
                    "evidence": f"Text contains: {', '.join(indicator['evidence'])}",
                    "clinical_significance": indicator["clinical_significance"],
                    "source": "text_analysis",
                }
            )

        # Add model-based indicators
        for indicator in model_indicators:
            combined.append(
                {
                    "factor": indicator["factor"],
                    "importance": indicator["importance"],
                    "evidence": indicator["evidence"],
                    "clinical_significance": "Identified by ML model analysis",
                    "source": "model_analysis",
                }
            )

        # Add context-based indicators
        if patient_context:
            if patient_context.get("previous_attempts", 0) > 0:
                combined.append(
                    {
                        "factor": "Previous Suicide Attempts",
                        "importance": 0.8,
                        "evidence": f"History of {patient_context['previous_attempts']} previous attempts",
                        "clinical_significance": "Prior attempts significantly increase future risk",
                        "source": "clinical_history",
                    }
                )

        # Sort by importance and remove duplicates
        combined = sorted(combined, key=lambda x: x["importance"], reverse=True)

        return combined

    def _combine_protective_factors(
        self, text_factors: List[Dict], model_factors: List[Dict], patient_context: Dict
    ) -> List[Dict[str, Any]]:
        """Combine protective factors from different sources."""

        combined = []

        # Add text-based protective factors
        for factor in text_factors:
            combined.append(
                {
                    "factor": factor["factor"].replace("_", " ").title(),
                    "strength": factor["weight"],
                    "evidence": f"Text mentions: {', '.join(factor['evidence'])}",
                    "clinical_significance": factor["clinical_significance"],
                }
            )

        # Add model-based protective factors
        for factor in model_factors:
            combined.append(factor)

        # Add context-based protective factors
        if patient_context:
            if patient_context.get("social_support", False):
                combined.append(
                    {
                        "factor": "Strong Social Support Network",
                        "strength": 0.7,
                        "evidence": "Documented social support system",
                        "clinical_significance": "Social support reduces suicide risk",
                    }
                )

        return sorted(combined, key=lambda x: x["strength"], reverse=True)

    def _identify_primary_concerns(self, risk_indicators: List[Dict], risk_level: str) -> List[str]:
        """Identify primary clinical concerns."""

        concerns = []

        if risk_level in ["high", "imminent"]:
            concerns.append(f"Patient assessed at {risk_level.upper()} suicide risk")

        # Add concerns based on top risk indicators
        top_indicators = risk_indicators[:3]  # Top 3

        for indicator in top_indicators:
            factor = indicator["factor"].lower()

            if "suicidal" in factor or "suicide" in factor:
                concerns.append("Active suicidal ideation expressed")
            elif "hopeless" in factor:
                concerns.append("Significant hopelessness and despair")
            elif "isolation" in factor or "alone" in factor:
                concerns.append("Social isolation and lack of support")
            elif "substance" in factor:
                concerns.append("Substance use affecting judgment and safety")

        return concerns[:5]  # Limit to top 5 concerns

    def _generate_recommendations(
        self,
        risk_level: str,
        risk_indicators: List[Dict],
        protective_factors: List[Dict],
        model_type: str,
    ) -> List[str]:
        """Generate clinical recommendations."""

        recommendations = []

        # Risk-level based recommendations
        if risk_level == "imminent":
            recommendations.extend(
                [
                    "IMMEDIATE: Implement continuous safety monitoring",
                    "IMMEDIATE: Remove all lethal means from environment",
                    "IMMEDIATE: Consider involuntary psychiatric evaluation",
                    "IMMEDIATE: Activate crisis response team",
                ]
            )
        elif risk_level == "high":
            recommendations.extend(
                [
                    "URGENT: Enhanced safety monitoring (q30min checks)",
                    "URGENT: Psychiatric evaluation within 4 hours",
                    "Develop comprehensive safety plan",
                    "Consider medication adjustment",
                ]
            )
        elif risk_level == "moderate":
            recommendations.extend(
                [
                    "Clinical review within 24 hours",
                    "Create or update safety plan",
                    "Consider therapy referral",
                    "Schedule follow-up within 1 week",
                ]
            )
        else:  # low risk
            recommendations.extend(
                [
                    "Routine follow-up as clinically indicated",
                    "Monitor for changes in risk factors",
                    "Reinforce protective factors",
                ]
            )

        # Factor-specific recommendations
        for indicator in risk_indicators[:3]:
            factor = indicator["factor"].lower()

            if "hopeless" in factor:
                recommendations.append("Address hopelessness with cognitive interventions")
            elif "isolation" in factor:
                recommendations.append("Activate social support network")
            elif "substance" in factor:
                recommendations.append("Assess substance use, consider treatment referral")
            elif "sleep" in factor:
                recommendations.append("Address sleep disturbance")

        # Model-specific recommendations
        if model_type == "bert":
            recommendations.append("Consider text-based monitoring of patient communications")

        return recommendations[:8]  # Limit to top 8 recommendations

    def _calculate_confidence(self, prediction: float, risk_indicators: List[Dict]) -> float:
        """Calculate explanation confidence."""

        # Higher confidence if:
        # 1. Prediction is not near decision boundaries
        # 2. Multiple consistent risk indicators
        # 3. Clear evidence in text

        boundary_distance = min(abs(prediction - 0.3), abs(prediction - 0.7), abs(prediction - 0.9))

        indicator_consistency = len(risk_indicators) / 10.0  # Normalize to 0-1

        confidence = 0.5 + (boundary_distance * 0.3) + (indicator_consistency * 0.2)

        return min(confidence, 1.0)

    def _calculate_uncertainty(self, prediction: float, model_type: str) -> float:
        """Calculate prediction uncertainty."""

        # Uncertainty is higher:
        # 1. Near decision boundaries
        # 2. For certain model types
        # 3. When clinical indicators are mixed

        boundary_uncertainty = 1 - min(
            abs(prediction - 0.3), abs(prediction - 0.7), abs(prediction - 0.9)
        )

        # Model-specific uncertainty
        model_uncertainty = {
            "svm": 0.2,  # More interpretable, lower uncertainty
            "bilstm": 0.3,  # Moderate uncertainty
            "bert": 0.4,  # Black box, higher uncertainty
        }.get(model_type, 0.3)

        total_uncertainty = (boundary_uncertainty + model_uncertainty) / 2

        return min(total_uncertainty, 1.0)

    def _find_similar_cases(self, text_indicators: Dict, prediction: float) -> Optional[Dict]:
        """Find similar cases (placeholder for case-based reasoning)."""

        # This would implement case-based reasoning in practice
        # For now, return a placeholder

        return {
            "similar_cases_found": 0,
            "note": "Case-based similarity matching not yet implemented",
        }

    def _get_relevant_literature(self, risk_indicators: List[Dict]) -> List[str]:
        """Get relevant literature references."""

        relevant_refs = []

        # Always include general references
        relevant_refs.append(self.literature_base[0])  # Meta-analysis
        relevant_refs.append(self.literature_base[4])  # Safety planning

        # Add specific references based on indicators
        for indicator in risk_indicators[:3]:
            factor = indicator["factor"].lower()

            if "hopeless" in factor:
                relevant_refs.append(self.literature_base[5])  # Joiner theory
            elif "social" in factor or "isolation" in factor:
                relevant_refs.append(self.literature_base[6])  # Interpersonal theory

        return relevant_refs[:3]  # Limit to top 3

    def _identify_limitations(
        self, model_type: str, text: str, patient_context: Optional[Dict]
    ) -> List[str]:
        """Identify model limitations and considerations."""

        limitations = []

        # General AI limitations
        limitations.append("AI model should supplement, not replace, clinical judgment")
        limitations.append("Model trained on social media data; clinical context may differ")

        # Model-specific limitations
        if model_type == "svm":
            limitations.append("SVM model relies on word patterns; may miss context")
        elif model_type == "bilstm":
            limitations.append("BiLSTM model may not capture long-range dependencies")
        elif model_type == "bert":
            limitations.append("BERT model is complex; decision process less interpretable")

        # Text-specific limitations
        if len(text.split()) < 10:
            limitations.append("Very short text may not provide sufficient information")

        # Context limitations
        if not patient_context:
            limitations.append("No clinical history available; prediction based on text only")

        return limitations[:5]  # Limit to top 5 most relevant
