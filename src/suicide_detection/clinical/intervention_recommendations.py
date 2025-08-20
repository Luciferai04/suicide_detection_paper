"""
Intervention Recommendations and Care Plan Generation

Provides evidence-based intervention recommendations, safety planning,
and FHIR-compatible care plan generation for suicide risk management.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .risk_stratification import RiskAssessment, RiskLevel, SeverityLevel

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of clinical interventions."""

    IMMEDIATE_SAFETY = "immediate_safety"
    CRISIS_INTERVENTION = "crisis_intervention"
    SAFETY_PLANNING = "safety_planning"
    THERAPEUTIC = "therapeutic"
    PHARMACOLOGICAL = "pharmacological"
    PSYCHOSOCIAL = "psychosocial"
    FOLLOW_UP = "follow_up"
    FAMILY_SUPPORT = "family_support"


class UrgencyLevel(Enum):
    """Urgency levels for interventions."""

    IMMEDIATE = "immediate"  # Within 1 hour
    URGENT = "urgent"  # Within 4 hours
    PRIORITY = "priority"  # Within 24 hours
    ROUTINE = "routine"  # Within 1 week


class AlertType(Enum):
    """Types of clinical alerts."""

    CRISIS_TEAM = "crisis_team"
    PSYCHIATRIST_ON_CALL = "psychiatrist_on_call"
    PRIMARY_CLINICIAN = "primary_clinician"
    EMERGENCY_SERVICES = "emergency_services"
    FAMILY_NOTIFICATION = "family_notification"
    SAFETY_MONITORING = "safety_monitoring"


@dataclass
class Intervention:
    """Single intervention recommendation."""

    intervention_id: str
    intervention_type: InterventionType
    urgency_level: UrgencyLevel

    # Intervention details
    title: str
    description: str
    rationale: str
    instructions: List[str]

    # Clinical parameters
    target_symptoms: List[str]
    expected_outcomes: List[str]
    contraindications: List[str]

    # Timing and duration
    recommended_start: datetime
    estimated_duration: Optional[timedelta] = None
    reassessment_interval: Optional[timedelta] = None

    # Evidence base
    evidence_level: str = "B"  # A, B, C evidence levels
    clinical_guidelines: List[str] = None

    # Implementation
    required_resources: List[str] = None
    staff_qualifications: List[str] = None

    def __post_init__(self):
        if self.clinical_guidelines is None:
            self.clinical_guidelines = []
        if self.required_resources is None:
            self.required_resources = []
        if self.staff_qualifications is None:
            self.staff_qualifications = []


@dataclass
class ClinicalAlert:
    """Clinical alert for healthcare team."""

    alert_id: str
    alert_type: AlertType
    urgency_level: UrgencyLevel

    message: str
    recipient_role: str
    patient_identifiers: Dict[str, str]

    created_timestamp: datetime
    acknowledgment_required: bool = True
    escalation_time: Optional[datetime] = None

    # Context
    risk_assessment: Optional[Dict[str, Any]] = None
    recommended_actions: List[str] = None

    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []


@dataclass
class SafetyPlan:
    """Comprehensive safety plan for suicide prevention."""

    plan_id: str
    patient_id: str
    created_date: datetime
    last_updated: datetime

    # Warning signs and triggers
    warning_signs: List[str]
    personal_triggers: List[str]
    environmental_triggers: List[str]

    # Coping strategies
    internal_coping_strategies: List[str]
    social_contacts_distraction: List[Dict[str, str]]  # name, phone, relationship
    family_friends_help: List[Dict[str, str]]
    professional_contacts: List[Dict[str, str]]

    # Environmental safety
    means_restriction: List[str]
    safe_environment_steps: List[str]

    # Crisis resources
    crisis_hotlines: List[Dict[str, str]]
    emergency_contacts: List[Dict[str, str]]
    preferred_emergency_location: str

    # Follow-up plan
    scheduled_appointments: List[Dict[str, Any]]
    self_monitoring_plan: List[str]

    # Validation
    patient_agreement: bool = False
    clinician_signature: Optional[str] = None
    witness_signature: Optional[str] = None


@dataclass
class CarePlan:
    """FHIR-compatible care plan."""

    # FHIR resource identifiers
    resource_type: str = "CarePlan"
    id: str = None
    status: str = "active"  # draft, active, on-hold, revoked, completed
    intent: str = "plan"  # proposal, plan, order

    # Patient and care team
    subject_reference: str = None  # Patient ID
    author_reference: str = None  # Clinician ID
    care_team: List[Dict[str, str]] = None

    # Clinical context
    category: List[str] = None
    title: str = None
    description: str = None
    created: datetime = None

    # Goals and activities
    goals: List[Dict[str, Any]] = None
    activities: List[Dict[str, Any]] = None

    # Addresses (conditions/problems)
    addresses: List[str] = None

    # Supporting information
    supporting_info: List[str] = None
    notes: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created is None:
            self.created = datetime.now()
        if self.care_team is None:
            self.care_team = []
        if self.category is None:
            self.category = ["assess-plan"]
        if self.goals is None:
            self.goals = []
        if self.activities is None:
            self.activities = []
        if self.addresses is None:
            self.addresses = []
        if self.supporting_info is None:
            self.supporting_info = []
        if self.notes is None:
            self.notes = []


class InterventionEngine:
    """
    Evidence-based intervention recommendation engine for suicide risk management.

    Implements clinical guidelines from:
    - American Psychiatric Association Practice Guidelines
    - National Suicide Prevention Guidelines
    - Joint Commission Standards
    - VA/DoD Clinical Practice Guidelines
    """

    def __init__(self):
        """Initialize intervention engine with evidence-based recommendations."""

        self.intervention_protocols = self._initialize_protocols()
        self.safety_plan_templates = self._initialize_safety_plan_templates()
        self.crisis_response_procedures = self._initialize_crisis_procedures()

        logger.info("Initialized evidence-based intervention engine")

    def generate_interventions(
        self,
        risk_assessment: RiskAssessment,
        patient_context: Dict[str, Any],
        care_team_preferences: Optional[Dict[str, Any]] = None,
    ) -> List[Intervention]:
        """
        Generate evidence-based intervention recommendations.

        Args:
            risk_assessment: Comprehensive risk assessment
            patient_context: Patient clinical and social context
            care_team_preferences: Care team preferences and constraints

        Returns:
            List of prioritized intervention recommendations
        """

        interventions = []

        # Immediate safety interventions based on risk level
        if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.IMMINENT]:
            interventions.extend(
                self._generate_immediate_safety_interventions(risk_assessment, patient_context)
            )

        # Crisis intervention protocols
        if risk_assessment.crisis_contact_indicated:
            interventions.extend(
                self._generate_crisis_interventions(risk_assessment, patient_context)
            )

        # Safety planning (all risk levels)
        interventions.append(
            self._generate_safety_planning_intervention(risk_assessment, patient_context)
        )

        # Therapeutic interventions
        interventions.extend(
            self._generate_therapeutic_interventions(risk_assessment, patient_context)
        )

        # Pharmacological considerations
        if risk_assessment.severity_level != SeverityLevel.MILD:
            interventions.extend(
                self._generate_pharmacological_interventions(risk_assessment, patient_context)
            )

        # Psychosocial interventions
        interventions.extend(
            self._generate_psychosocial_interventions(risk_assessment, patient_context)
        )

        # Follow-up planning
        interventions.append(
            self._generate_follow_up_intervention(risk_assessment, patient_context)
        )

        # Sort by urgency and clinical priority
        interventions.sort(key=lambda x: (x.urgency_level.value, x.intervention_type.value))

        logger.info(f"Generated {len(interventions)} intervention recommendations")
        return interventions

    def generate_clinical_alerts(
        self,
        risk_assessment: RiskAssessment,
        interventions: List[Intervention],
        patient_context: Dict[str, Any],
    ) -> List[ClinicalAlert]:
        """Generate clinical alerts for care team."""

        alerts = []
        patient_id = patient_context.get("patient_id", "unknown")

        # High/imminent risk alerts
        if risk_assessment.risk_level == RiskLevel.IMMINENT:
            alerts.append(
                ClinicalAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.CRISIS_TEAM,
                    urgency_level=UrgencyLevel.IMMEDIATE,
                    message=f"IMMEDIATE CRISIS ALERT: Patient {patient_id} assessed at IMMINENT suicide risk",
                    recipient_role="crisis_team",
                    patient_identifiers={"id": patient_id},
                    created_timestamp=datetime.now(),
                    risk_assessment=asdict(risk_assessment),
                    recommended_actions=[
                        "Immediate safety assessment required",
                        "Consider involuntary hold if necessary",
                        "Remove lethal means",
                        "Continuous monitoring",
                        "Crisis team response within 1 hour",
                    ],
                )
            )

            alerts.append(
                ClinicalAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.PSYCHIATRIST_ON_CALL,
                    urgency_level=UrgencyLevel.IMMEDIATE,
                    message="Urgent psychiatric consultation needed - IMMINENT suicide risk",
                    recipient_role="psychiatrist",
                    patient_identifiers={"id": patient_id},
                    created_timestamp=datetime.now(),
                    escalation_time=datetime.now() + timedelta(hours=1),
                )
            )

        elif risk_assessment.risk_level == RiskLevel.HIGH:
            alerts.append(
                ClinicalAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.PSYCHIATRIST_ON_CALL,
                    urgency_level=UrgencyLevel.URGENT,
                    message="HIGH suicide risk assessment - urgent evaluation needed",
                    recipient_role="psychiatrist",
                    patient_identifiers={"id": patient_id},
                    created_timestamp=datetime.now(),
                    recommended_actions=[
                        "Psychiatric evaluation within 4 hours",
                        "Safety plan development",
                        "Consider medication adjustment",
                        "Increase monitoring frequency",
                    ],
                )
            )

        # Primary clinician notification for moderate risk
        elif risk_assessment.risk_level == RiskLevel.MODERATE:
            alerts.append(
                ClinicalAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.PRIMARY_CLINICIAN,
                    urgency_level=UrgencyLevel.PRIORITY,
                    message="MODERATE suicide risk - clinical review recommended",
                    recipient_role="primary_clinician",
                    patient_identifiers={"id": patient_id},
                    created_timestamp=datetime.now(),
                    recommended_actions=[
                        "Clinical review within 24 hours",
                        "Update safety plan",
                        "Consider therapy referral",
                        "Schedule follow-up",
                    ],
                )
            )

        # Family notification if high risk and consent available
        if risk_assessment.risk_level in [
            RiskLevel.HIGH,
            RiskLevel.IMMINENT,
        ] and patient_context.get("family_notification_consent", False):
            alerts.append(
                ClinicalAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.FAMILY_NOTIFICATION,
                    urgency_level=UrgencyLevel.URGENT,
                    message="Family notification recommended due to high suicide risk",
                    recipient_role="care_coordinator",
                    patient_identifiers={"id": patient_id},
                    created_timestamp=datetime.now(),
                )
            )

        return alerts

    def create_safety_plan(
        self,
        risk_assessment: RiskAssessment,
        patient_context: Dict[str, Any],
        existing_plan: Optional[SafetyPlan] = None,
    ) -> SafetyPlan:
        """Create or update comprehensive safety plan."""

        patient_id = patient_context.get("patient_id", str(uuid.uuid4()))

        if existing_plan:
            # Update existing plan
            safety_plan = existing_plan
            safety_plan.last_updated = datetime.now()
        else:
            # Create new plan
            safety_plan = SafetyPlan(
                plan_id=str(uuid.uuid4()),
                patient_id=patient_id,
                created_date=datetime.now(),
                last_updated=datetime.now(),
                warning_signs=[],
                personal_triggers=[],
                environmental_triggers=[],
                internal_coping_strategies=[],
                social_contacts_distraction=[],
                family_friends_help=[],
                professional_contacts=[],
                means_restriction=[],
                safe_environment_steps=[],
                crisis_hotlines=[],
                emergency_contacts=[],
                preferred_emergency_location="",
                scheduled_appointments=[],
                self_monitoring_plan=[],
            )

        # Populate based on risk assessment
        self._populate_safety_plan_from_assessment(safety_plan, risk_assessment, patient_context)

        return safety_plan

    def _initialize_protocols(self) -> Dict[str, Dict]:
        """Initialize evidence-based intervention protocols."""

        return {
            "immediate_safety": {
                "imminent_risk": [
                    "Continuous visual monitoring",
                    "Remove all potentially lethal means",
                    "One-to-one observation",
                    "Consider involuntary psychiatric hold",
                    "Crisis team activation",
                ],
                "high_risk": [
                    "Frequent check-ins (every 15-30 minutes)",
                    "Secure environment assessment",
                    "Limited access to means",
                    "Safety contract if appropriate",
                    "Crisis plan activation",
                ],
            },
            "therapeutic": {
                "evidence_based_therapies": [
                    "Cognitive Behavioral Therapy for Suicide Prevention (CBT-SP)",
                    "Dialectical Behavior Therapy (DBT)",
                    "Collaborative Assessment and Management of Suicidality (CAMS)",
                    "Safety Planning Intervention (SPI)",
                    "Brief Intervention and Contact (BIC)",
                ],
                "crisis_interventions": [
                    "Crisis Response Plan (CRP)",
                    "Mobile Crisis Team deployment",
                    "Peer support specialist contact",
                    "Family crisis intervention",
                ],
            },
            "pharmacological": {
                "acute_interventions": [
                    "Clozapine for treatment-resistant cases with psychosis",
                    "Lithium for bipolar disorder with suicide risk",
                    "Ketamine/esketamine for treatment-resistant depression",
                    "Avoid tricyclics and barbiturates in high-risk patients",
                ],
                "maintenance": [
                    "Antidepressant optimization",
                    "Mood stabilizer adjustment",
                    "Antipsychotic consideration for severe cases",
                    "Medication adherence support",
                ],
            },
        }

    def _initialize_safety_plan_templates(self) -> Dict[str, List]:
        """Initialize safety plan templates."""

        return {
            "warning_signs": [
                "Feeling hopeless or trapped",
                "Increase in suicidal thoughts",
                "Feeling unbearable emotional pain",
                "Social withdrawal and isolation",
                "Sleep disturbances",
                "Increased substance use",
                "Feeling like a burden to others",
            ],
            "coping_strategies": [
                "Deep breathing exercises",
                "Progressive muscle relaxation",
                "Mindfulness meditation",
                "Physical exercise or walking",
                "Listening to music",
                "Writing in a journal",
                "Engaging in hobbies",
            ],
            "crisis_hotlines": [
                {
                    "name": "National Suicide Prevention Lifeline",
                    "phone": "988",
                    "available": "24/7",
                },
                {"name": "Crisis Text Line", "phone": "Text HOME to 741741", "available": "24/7"},
            ],
        }

    def _initialize_crisis_procedures(self) -> Dict[str, List]:
        """Initialize crisis response procedures."""

        return {
            "imminent_risk": [
                "Call 911 if immediate danger",
                "Activate crisis team within 1 hour",
                "Consider involuntary psychiatric evaluation",
                "Notify psychiatrist on-call immediately",
                "Ensure continuous monitoring",
            ],
            "high_risk": [
                "Contact crisis services within 4 hours",
                "Schedule urgent psychiatric evaluation",
                "Implement safety plan immediately",
                "Increase monitoring frequency",
                "Consider partial hospitalization",
            ],
            "moderate_risk": [
                "Schedule clinical review within 24 hours",
                "Update or create safety plan",
                "Consider outpatient therapy referral",
                "Plan follow-up within 1 week",
                "Engage support system",
            ],
        }

    def _generate_immediate_safety_interventions(
        self, risk_assessment: RiskAssessment, patient_context: Dict[str, Any]
    ) -> List[Intervention]:
        """Generate immediate safety interventions."""

        interventions = []

        if risk_assessment.risk_level == RiskLevel.IMMINENT:
            interventions.append(
                Intervention(
                    intervention_id=str(uuid.uuid4()),
                    intervention_type=InterventionType.IMMEDIATE_SAFETY,
                    urgency_level=UrgencyLevel.IMMEDIATE,
                    title="Continuous Safety Monitoring",
                    description="Implement immediate continuous safety monitoring protocol",
                    rationale=f"Patient assessed at IMMINENT risk (confidence: {risk_assessment.prediction_confidence:.2f})",
                    instructions=[
                        "Continuous visual monitoring",
                        "Remove all potentially lethal means from environment",
                        "One-to-one observation if in facility",
                        "Document safety checks every 15 minutes",
                        "Activate crisis response team",
                    ],
                    target_symptoms=["Suicidal ideation", "Self-harm risk"],
                    expected_outcomes=["Immediate safety secured", "Risk stabilization"],
                    contraindications=[],
                    recommended_start=datetime.now(),
                    evidence_level="A",
                    clinical_guidelines=["Joint Commission Standards", "APA Practice Guidelines"],
                    required_resources=["Trained monitoring staff", "Secure environment"],
                    staff_qualifications=[
                        "Mental health training",
                        "Crisis intervention certification",
                    ],
                )
            )

        elif risk_assessment.risk_level == RiskLevel.HIGH:
            interventions.append(
                Intervention(
                    intervention_id=str(uuid.uuid4()),
                    intervention_type=InterventionType.IMMEDIATE_SAFETY,
                    urgency_level=UrgencyLevel.URGENT,
                    title="Enhanced Safety Monitoring",
                    description="Implement enhanced safety monitoring with frequent assessments",
                    rationale="High suicide risk requires increased monitoring and safety measures",
                    instructions=[
                        "Safety checks every 30 minutes",
                        "Secure potentially dangerous items",
                        "Assess environment for safety risks",
                        "Implement safety contract if clinically appropriate",
                        "Prepare crisis intervention plan",
                    ],
                    target_symptoms=["Suicidal ideation", "Impulsivity"],
                    expected_outcomes=["Risk reduction", "Safety plan implementation"],
                    contraindications=["Patient actively psychotic"],
                    recommended_start=datetime.now(),
                    evidence_level="B",
                    clinical_guidelines=["APA Practice Guidelines"],
                    required_resources=["Clinical staff", "Safety assessment tools"],
                )
            )

        return interventions

    def _generate_crisis_interventions(
        self, risk_assessment: RiskAssessment, patient_context: Dict[str, Any]
    ) -> List[Intervention]:
        """Generate crisis intervention recommendations."""

        interventions = []

        interventions.append(
            Intervention(
                intervention_id=str(uuid.uuid4()),
                intervention_type=InterventionType.CRISIS_INTERVENTION,
                urgency_level=UrgencyLevel.URGENT,
                title="Crisis Team Activation",
                description="Activate specialized crisis response team for comprehensive evaluation",
                rationale="Crisis-level risk requires specialized multidisciplinary response",
                instructions=[
                    "Contact crisis team within 1 hour for imminent risk, 4 hours for high risk",
                    "Prepare comprehensive risk assessment documentation",
                    "Coordinate with psychiatrist on-call",
                    "Implement crisis stabilization protocol",
                    "Consider emergency psychiatric evaluation",
                ],
                target_symptoms=risk_assessment.acute_risk_factors,
                expected_outcomes=["Crisis stabilization", "Risk reduction", "Safety planning"],
                contraindications=[],
                recommended_start=datetime.now(),
                evidence_level="A",
                clinical_guidelines=["National Suicide Prevention Guidelines"],
                required_resources=["Crisis response team", "Psychiatric consultation"],
                staff_qualifications=["Crisis intervention specialists", "Licensed clinicians"],
            )
        )

        return interventions

    def _generate_safety_planning_intervention(
        self, risk_assessment: RiskAssessment, patient_context: Dict[str, Any]
    ) -> Intervention:
        """Generate safety planning intervention."""

        urgency = (
            UrgencyLevel.IMMEDIATE
            if risk_assessment.risk_level == RiskLevel.IMMINENT
            else UrgencyLevel.URGENT
        )

        return Intervention(
            intervention_id=str(uuid.uuid4()),
            intervention_type=InterventionType.SAFETY_PLANNING,
            urgency_level=urgency,
            title="Comprehensive Safety Plan Development",
            description="Develop or update comprehensive safety plan with patient collaboration",
            rationale="Safety planning is evidence-based intervention for suicide risk reduction",
            instructions=[
                "Identify personal warning signs and triggers",
                "Develop coping strategies and support contacts",
                "Implement means restriction plan",
                "Establish crisis contact protocols",
                "Schedule follow-up appointments",
                "Obtain patient agreement and signatures",
            ],
            target_symptoms=["Suicidal ideation", "Crisis vulnerability"],
            expected_outcomes=[
                "Increased safety awareness",
                "Crisis coping skills",
                "Support activation",
            ],
            contraindications=["Severe cognitive impairment"],
            recommended_start=datetime.now(),
            estimated_duration=timedelta(hours=1),
            reassessment_interval=timedelta(days=7),
            evidence_level="A",
            clinical_guidelines=["Safety Planning Intervention"],
            required_resources=["Safety plan template", "Clinical time"],
            staff_qualifications=["Mental health clinician", "Safety planning training"],
        )

    def _generate_therapeutic_interventions(
        self, risk_assessment: RiskAssessment, patient_context: Dict[str, Any]
    ) -> List[Intervention]:
        """Generate evidence-based therapeutic interventions."""

        interventions = []

        # Evidence-based therapy referral
        interventions.append(
            Intervention(
                intervention_id=str(uuid.uuid4()),
                intervention_type=InterventionType.THERAPEUTIC,
                urgency_level=UrgencyLevel.PRIORITY,
                title="Evidence-Based Therapy Referral",
                description="Refer for specialized suicide-focused therapy",
                rationale="Evidence-based psychotherapy reduces suicide risk and improves outcomes",
                instructions=[
                    "Refer for Cognitive Behavioral Therapy for Suicide Prevention (CBT-SP)",
                    "Consider Dialectical Behavior Therapy (DBT) for recurrent risk",
                    "Evaluate for Collaborative Assessment and Management of Suicidality (CAMS)",
                    "Coordinate with primary mental health provider",
                    "Schedule initial appointment within 1 week",
                ],
                target_symptoms=["Suicidal ideation", "Hopelessness", "Emotion dysregulation"],
                expected_outcomes=[
                    "Reduced suicidal ideation",
                    "Improved coping",
                    "Better emotional regulation",
                ],
                contraindications=["Active psychosis without stabilization"],
                recommended_start=datetime.now() + timedelta(days=1),
                estimated_duration=timedelta(weeks=12),
                reassessment_interval=timedelta(weeks=4),
                evidence_level="A",
                clinical_guidelines=["APA Practice Guidelines", "Cochrane Reviews"],
                required_resources=["Trained therapist", "Therapy space"],
                staff_qualifications=[
                    "Licensed mental health professional",
                    "Suicide-specific training",
                ],
            )
        )

        return interventions

    def _generate_pharmacological_interventions(
        self, risk_assessment: RiskAssessment, patient_context: Dict[str, Any]
    ) -> List[Intervention]:
        """Generate pharmacological intervention recommendations."""

        interventions = []

        interventions.append(
            Intervention(
                intervention_id=str(uuid.uuid4()),
                intervention_type=InterventionType.PHARMACOLOGICAL,
                urgency_level=UrgencyLevel.URGENT,
                title="Psychiatric Medication Review",
                description="Comprehensive review and optimization of psychiatric medications",
                rationale="Medication optimization can significantly reduce suicide risk",
                instructions=[
                    "Review current psychiatric medications for efficacy and safety",
                    "Consider lithium for bipolar disorder with suicide risk",
                    "Evaluate clozapine for treatment-resistant cases",
                    "Consider ketamine/esketamine for severe treatment-resistant depression",
                    "Avoid tricyclics and barbiturates in high-risk patients",
                    "Optimize antidepressant therapy",
                    "Address medication adherence barriers",
                ],
                target_symptoms=[
                    "Depression",
                    "Bipolar disorder",
                    "Psychosis",
                    "Treatment resistance",
                ],
                expected_outcomes=["Symptom improvement", "Risk reduction", "Better adherence"],
                contraindications=["Medication allergies", "Drug interactions"],
                recommended_start=datetime.now() + timedelta(hours=4),
                reassessment_interval=timedelta(weeks=2),
                evidence_level="A",
                clinical_guidelines=["APA Practice Guidelines", "VA/DoD Guidelines"],
                required_resources=["Psychiatrist", "Pharmacy consultation"],
                staff_qualifications=["Board-certified psychiatrist"],
            )
        )

        return interventions

    def _generate_psychosocial_interventions(
        self, risk_assessment: RiskAssessment, patient_context: Dict[str, Any]
    ) -> List[Intervention]:
        """Generate psychosocial support interventions."""

        interventions = []

        interventions.append(
            Intervention(
                intervention_id=str(uuid.uuid4()),
                intervention_type=InterventionType.PSYCHOSOCIAL,
                urgency_level=UrgencyLevel.PRIORITY,
                title="Social Support Network Activation",
                description="Activate and strengthen patient's social support network",
                rationale="Social support is a key protective factor against suicide risk",
                instructions=[
                    "Identify key support persons in patient's life",
                    "Assess family dynamics and support capacity",
                    "Connect with peer support services if available",
                    "Address social isolation and loneliness",
                    "Consider family therapy or support groups",
                    "Evaluate need for case management services",
                ],
                target_symptoms=["Social isolation", "Loneliness", "Lack of support"],
                expected_outcomes=[
                    "Increased social connections",
                    "Better support utilization",
                    "Reduced isolation",
                ],
                contraindications=["Abusive relationships"],
                recommended_start=datetime.now() + timedelta(hours=24),
                evidence_level="B",
                clinical_guidelines=["National Suicide Prevention Guidelines"],
                required_resources=["Social worker", "Family involvement"],
                staff_qualifications=["Licensed social worker", "Family therapist"],
            )
        )

        return interventions

    def _generate_follow_up_intervention(
        self, risk_assessment: RiskAssessment, patient_context: Dict[str, Any]
    ) -> Intervention:
        """Generate follow-up intervention plan."""

        # Follow-up timing based on risk level
        if risk_assessment.risk_level == RiskLevel.IMMINENT:
            follow_up_time = timedelta(hours=4)
            urgency = UrgencyLevel.IMMEDIATE
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            follow_up_time = timedelta(hours=24)
            urgency = UrgencyLevel.URGENT
        elif risk_assessment.risk_level == RiskLevel.MODERATE:
            follow_up_time = timedelta(days=3)
            urgency = UrgencyLevel.PRIORITY
        else:
            follow_up_time = timedelta(days=7)
            urgency = UrgencyLevel.ROUTINE

        return Intervention(
            intervention_id=str(uuid.uuid4()),
            intervention_type=InterventionType.FOLLOW_UP,
            urgency_level=urgency,
            title="Structured Follow-up Plan",
            description="Implement structured follow-up monitoring and care coordination",
            rationale="Regular follow-up reduces suicide risk and improves treatment engagement",
            instructions=[
                f"Schedule follow-up appointment within {follow_up_time}",
                "Conduct structured risk reassessment",
                "Review safety plan effectiveness",
                "Monitor treatment adherence and response",
                "Coordinate with all providers in care team",
                "Document progress and any risk changes",
            ],
            target_symptoms=["Ongoing risk monitoring", "Treatment engagement"],
            expected_outcomes=[
                "Continued risk reduction",
                "Treatment adherence",
                "Early intervention",
            ],
            contraindications=[],
            recommended_start=datetime.now() + follow_up_time,
            reassessment_interval=follow_up_time,
            evidence_level="B",
            clinical_guidelines=["National Suicide Prevention Guidelines"],
            required_resources=["Clinical appointment", "Care coordination"],
            staff_qualifications=["Licensed clinician"],
        )

    def _populate_safety_plan_from_assessment(
        self,
        safety_plan: SafetyPlan,
        risk_assessment: RiskAssessment,
        patient_context: Dict[str, Any],
    ):
        """Populate safety plan based on risk assessment."""

        # Warning signs from risk factors
        safety_plan.warning_signs.extend(
            [
                "Increased hopelessness",
                "Sleep disturbances",
                "Social withdrawal",
                "Increased substance use",
            ]
        )

        # Triggers from acute risk factors
        safety_plan.personal_triggers.extend(risk_assessment.acute_risk_factors[:3])

        # Standard coping strategies
        safety_plan.internal_coping_strategies.extend(
            [
                "Deep breathing exercises",
                "Call a friend or family member",
                "Go for a walk",
                "Listen to music",
                "Practice mindfulness",
            ]
        )

        # Crisis contacts
        safety_plan.crisis_hotlines.extend(
            [
                {
                    "name": "National Suicide Prevention Lifeline",
                    "phone": "988",
                    "available": "24/7",
                },
                {"name": "Crisis Text Line", "phone": "Text HOME to 741741", "available": "24/7"},
            ]
        )

        # Emergency location
        safety_plan.preferred_emergency_location = patient_context.get(
            "preferred_hospital", "Nearest Emergency Department"
        )


class CarePlanGenerator:
    """FHIR-compatible care plan generator for suicide risk management."""

    def __init__(self):
        """Initialize care plan generator."""
        logger.info("Initialized FHIR care plan generator")

    def generate_care_plan(
        self,
        risk_assessment: RiskAssessment,
        interventions: List[Intervention],
        safety_plan: SafetyPlan,
        patient_context: Dict[str, Any],
    ) -> CarePlan:
        """Generate comprehensive FHIR-compatible care plan."""

        care_plan = CarePlan()

        # Basic metadata
        care_plan.subject_reference = f"Patient/{patient_context.get('patient_id', 'unknown')}"
        care_plan.author_reference = (
            f"Practitioner/{patient_context.get('clinician_id', 'unknown')}"
        )
        care_plan.title = (
            f"Suicide Risk Management Plan - {risk_assessment.risk_level.value.upper()} Risk"
        )
        care_plan.description = f"Comprehensive care plan for suicide risk management (Risk Level: {risk_assessment.risk_level.value})"

        # Categories
        care_plan.category = ["assess-plan", "mental-health", "crisis-prevention"]

        # Goals
        care_plan.goals = [
            {
                "id": str(uuid.uuid4()),
                "description": "Reduce suicide risk to low/manageable level",
                "priority": "high-priority",
                "start": datetime.now().isoformat(),
                "target": [
                    {
                        "measure": "Suicide risk level",
                        "detail": "Reduce from current level to LOW within 30 days",
                    }
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "description": "Improve safety planning and coping skills",
                "priority": "medium-priority",
                "start": datetime.now().isoformat(),
            },
        ]

        # Activities from interventions
        for intervention in interventions:
            activity = {
                "id": intervention.intervention_id,
                "status": "not-started",
                "intent": "plan",
                "category": intervention.intervention_type.value,
                "description": intervention.description,
                "scheduled": intervention.recommended_start.isoformat(),
                "performer": intervention.staff_qualifications,
                "reason": intervention.rationale,
            }
            care_plan.activities.append(activity)

        # Addresses (conditions)
        care_plan.addresses.extend(
            ["Suicidal ideation", "Suicide risk", risk_assessment.severity_level.value]
        )

        # Supporting information
        care_plan.supporting_info.append(
            f"Risk Assessment ID: {risk_assessment.assessment_timestamp}"
        )
        care_plan.supporting_info.append(f"Safety Plan ID: {safety_plan.plan_id}")

        # Notes
        care_plan.notes.append(
            {
                "time": datetime.now().isoformat(),
                "text": f"Risk level: {risk_assessment.risk_level.value}, "
                f"Confidence: {risk_assessment.prediction_confidence:.2f}, "
                f"Crisis contact indicated: {risk_assessment.crisis_contact_indicated}",
            }
        )

        return care_plan

    def export_fhir_json(self, care_plan: CarePlan) -> str:
        """Export care plan as FHIR JSON."""

        fhir_dict = asdict(care_plan)

        # Convert datetime objects to ISO strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        fhir_dict = convert_datetime(fhir_dict)

        return json.dumps(fhir_dict, indent=2)
