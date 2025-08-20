"""
Healthcare Integration and FHIR/HL7 Connectivity

Provides healthcare system integration capabilities including:
- FHIR R4 resource management and exchange
- HL7 v2 message processing for legacy systems
- EHR interface and API connectivity
- Clinical decision support integration
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

from .intervention_recommendations import CarePlan, ClinicalAlert
from .risk_stratification import RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """HL7 v2 message types."""

    ADT_A01 = "ADT^A01"  # Admit/visit notification
    ADT_A03 = "ADT^A03"  # Discharge/end visit
    ADT_A08 = "ADT^A08"  # Update patient information
    ORM_O01 = "ORM^O01"  # General order message
    ORU_R01 = "ORU^R01"  # Observation result
    BAR_P01 = "BAR^P01"  # Add patient account
    SIU_S12 = "SIU^S12"  # Notification of new appointment booking


class FHIRVersion(Enum):
    """Supported FHIR versions."""

    R4 = "4.0.1"
    R5 = "5.0.0"


class IntegrationStatus(Enum):
    """Integration connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"
    MAINTENANCE = "maintenance"


@dataclass
class EHRConfiguration:
    """EHR system configuration."""

    system_name: str
    system_type: str  # "epic", "cerner", "allscripts", "meditech", etc.
    version: str

    # Connection parameters
    base_url: str
    fhir_endpoint: Optional[str] = None
    hl7_endpoint: Optional[str] = None

    # Authentication
    auth_type: str = "oauth2"  # "oauth2", "basic", "api_key", "certificate"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    api_key: Optional[str] = None

    # FHIR capabilities
    fhir_version: FHIRVersion = FHIRVersion.R4
    supported_resources: List[str] = None

    # Security and compliance
    encryption_required: bool = True
    audit_logging: bool = True
    patient_consent_required: bool = True

    # Rate limiting
    requests_per_minute: int = 100
    timeout_seconds: int = 30

    def __post_init__(self):
        if self.supported_resources is None:
            self.supported_resources = [
                "Patient",
                "Practitioner",
                "Organization",
                "Encounter",
                "Observation",
                "DiagnosticReport",
                "CarePlan",
                "Goal",
                "Task",
                "Communication",
            ]


@dataclass
class HL7Message:
    """HL7 v2 message structure."""

    message_type: MessageType
    control_id: str
    timestamp: datetime

    # Message segments
    msh: Dict[str, Any]  # Message Header
    pid: Optional[Dict[str, Any]] = None  # Patient Identification
    pv1: Optional[Dict[str, Any]] = None  # Patient Visit
    obx: Optional[List[Dict[str, Any]]] = None  # Observation/Result
    orc: Optional[Dict[str, Any]] = None  # Common Order
    obr: Optional[Dict[str, Any]] = None  # Observation Request

    def __post_init__(self):
        if self.obx is None:
            self.obx = []


@dataclass
class ClinicalDecisionSupport:
    """Clinical decision support alert/recommendation."""

    alert_id: str
    patient_id: str
    clinician_id: str

    # Alert details
    alert_type: str  # "drug_interaction", "allergy", "suicide_risk", "clinical_reminder"
    severity: str  # "low", "medium", "high", "critical"
    title: str
    message: str

    # Clinical context
    triggers: List[str]
    recommendations: List[str]

    # Timing
    created_timestamp: datetime
    expiration_timestamp: Optional[datetime] = None

    # Actions
    actions_available: List[str] = None
    acknowledgment_required: bool = True

    def __post_init__(self):
        if self.actions_available is None:
            self.actions_available = ["acknowledge", "dismiss", "snooze", "override"]


class FHIRConnector:
    """
    FHIR R4 connector for healthcare system integration.

    Provides standardized healthcare data exchange using FHIR resources
    with focus on suicide risk assessment and care planning.
    """

    def __init__(self, config: EHRConfiguration):
        """
        Initialize FHIR connector with EHR configuration.

        Args:
            config: EHR system configuration
        """
        self.config = config
        self.session = requests.Session()
        self.access_token = None
        self.token_expires = None
        self.status = IntegrationStatus.DISCONNECTED

        # Set up session headers
        self.session.headers.update(
            {
                "Accept": "application/fhir+json",
                "Content-Type": "application/fhir+json",
                "User-Agent": "SuicideDetection-CDS/1.0.0",
            }
        )

        logger.info(f"Initialized FHIR connector for {config.system_name}")

    def connect(self) -> bool:
        """
        Establish connection to FHIR server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.status = IntegrationStatus.AUTHENTICATING

            # Authenticate based on configured method
            if self.config.auth_type == "oauth2":
                success = self._oauth2_authenticate()
            elif self.config.auth_type == "api_key":
                success = self._api_key_authenticate()
            else:
                logger.error(f"Unsupported auth type: {self.config.auth_type}")
                return False

            if success:
                # Test connection with capability statement
                capability_url = urljoin(self.config.fhir_endpoint, "metadata")
                response = self.session.get(capability_url, timeout=self.config.timeout_seconds)

                if response.status_code == 200:
                    self.status = IntegrationStatus.CONNECTED
                    logger.info(f"Successfully connected to {self.config.system_name}")
                    return True
                else:
                    logger.error(f"Failed to retrieve capability statement: {response.status_code}")

            self.status = IntegrationStatus.ERROR
            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.status = IntegrationStatus.ERROR
            return False

    def create_patient_resource(self, patient_data: Dict[str, Any]) -> Optional[str]:
        """
        Create FHIR Patient resource.

        Args:
            patient_data: Patient demographic and identifier data

        Returns:
            Patient resource ID if successful, None otherwise
        """

        patient_resource = {
            "resourceType": "Patient",
            "identifier": [
                {
                    "use": "usual",
                    "system": "http://hospital.smartplatforms.org",
                    "value": patient_data.get("mrn", str(uuid.uuid4())),
                }
            ],
            "name": [
                {
                    "use": "official",
                    "family": patient_data.get("last_name", "Unknown"),
                    "given": [patient_data.get("first_name", "Unknown")],
                }
            ],
            "gender": patient_data.get("gender", "unknown"),
            "birthDate": patient_data.get("birth_date"),
            "address": (
                [
                    {
                        "use": "home",
                        "city": patient_data.get("city"),
                        "state": patient_data.get("state"),
                        "postalCode": patient_data.get("zip_code"),
                    }
                ]
                if patient_data.get("city")
                else []
            ),
        }

        return self._create_resource("Patient", patient_resource)

    def create_risk_assessment_observation(
        self, patient_id: str, risk_assessment: RiskAssessment, encounter_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create FHIR Observation resource for suicide risk assessment.

        Args:
            patient_id: FHIR Patient resource ID
            risk_assessment: Risk assessment results
            encounter_id: Associated encounter ID

        Returns:
            Observation resource ID if successful
        """

        # Map risk level to standardized coding
        risk_level_coding = {
            RiskLevel.LOW: {"code": "low", "display": "Low Risk"},
            RiskLevel.MODERATE: {"code": "moderate", "display": "Moderate Risk"},
            RiskLevel.HIGH: {"code": "high", "display": "High Risk"},
            RiskLevel.IMMINENT: {"code": "critical", "display": "Imminent Risk"},
        }

        observation = {
            "resourceType": "Observation",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "72133-2",
                        "display": "Suicide risk assessment",
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": risk_assessment.assessment_timestamp.isoformat(),
            "valueCodeableConcept": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": risk_level_coding[risk_assessment.risk_level]["code"],
                        "display": risk_level_coding[risk_assessment.risk_level]["display"],
                    }
                ]
            },
            "component": [
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "LA6115-3",
                                "display": "Risk probability",
                            }
                        ]
                    },
                    "valueQuantity": {
                        "value": round(risk_assessment.risk_probability, 3),
                        "unit": "probability",
                        "system": "http://unitsofmeasure.org",
                    },
                },
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "LA6116-1",
                                "display": "Prediction confidence",
                            }
                        ]
                    },
                    "valueQuantity": {
                        "value": round(risk_assessment.prediction_confidence, 3),
                        "unit": "confidence",
                        "system": "http://unitsofmeasure.org",
                    },
                },
            ],
        }

        if encounter_id:
            observation["encounter"] = {"reference": f"Encounter/{encounter_id}"}

        return self._create_resource("Observation", observation)

    def create_care_plan_resource(self, patient_id: str, care_plan: CarePlan) -> Optional[str]:
        """
        Create FHIR CarePlan resource.

        Args:
            patient_id: FHIR Patient resource ID
            care_plan: Care plan object

        Returns:
            CarePlan resource ID if successful
        """

        # Convert our CarePlan to FHIR format
        fhir_care_plan = {
            "resourceType": "CarePlan",
            "status": care_plan.status,
            "intent": care_plan.intent,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                            "code": cat,
                            "display": cat.replace("-", " ").title(),
                        }
                    ]
                }
                for cat in care_plan.category
            ],
            "title": care_plan.title,
            "description": care_plan.description,
            "subject": {"reference": f"Patient/{patient_id}"},
            "created": care_plan.created.isoformat(),
            "goal": [
                {"reference": f"Goal/{goal.get('id', uuid.uuid4())}"} for goal in care_plan.goals
            ],
            "activity": [
                {
                    "detail": {
                        "status": activity.get("status", "not-started"),
                        "description": activity.get("description", ""),
                        "scheduledTiming": (
                            {"event": [activity.get("scheduled")]}
                            if activity.get("scheduled")
                            else None
                        ),
                    }
                }
                for activity in care_plan.activities
            ],
        }

        return self._create_resource("CarePlan", fhir_care_plan)

    def create_clinical_alert_task(
        self, patient_id: str, clinician_id: str, alert: ClinicalAlert
    ) -> Optional[str]:
        """
        Create FHIR Task resource for clinical alert.

        Args:
            patient_id: FHIR Patient resource ID
            clinician_id: FHIR Practitioner resource ID
            alert: Clinical alert

        Returns:
            Task resource ID if successful
        """

        # Map urgency to FHIR priority
        priority_mapping = {
            "immediate": "stat",
            "urgent": "asap",
            "priority": "urgent",
            "routine": "routine",
        }

        task = {
            "resourceType": "Task",
            "status": "requested",
            "intent": "order",
            "priority": priority_mapping.get(alert.urgency_level.value, "routine"),
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "72133-2",
                        "display": "Clinical decision support alert",
                    }
                ]
            },
            "description": alert.message,
            "for": {"reference": f"Patient/{patient_id}"},
            "requester": {"reference": "Device/suicide-risk-cds"},
            "owner": {"reference": f"Practitioner/{clinician_id}"},
            "authoredOn": alert.created_timestamp.isoformat(),
            "input": [
                {
                    "type": {"coding": [{"code": "alert-type", "display": "Alert Type"}]},
                    "valueString": alert.alert_type.value,
                }
            ],
        }

        if alert.escalation_time:
            task["restriction"] = {"period": {"end": alert.escalation_time.isoformat()}}

        return self._create_resource("Task", task)

    def search_patient(
        self,
        identifier: Optional[str] = None,
        family_name: Optional[str] = None,
        given_name: Optional[str] = None,
        birth_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for patients using FHIR search parameters.

        Args:
            identifier: Patient identifier (MRN, etc.)
            family_name: Patient family name
            given_name: Patient given name
            birth_date: Patient birth date (YYYY-MM-DD)

        Returns:
            List of matching patient resources
        """

        search_params = {}
        if identifier:
            search_params["identifier"] = identifier
        if family_name:
            search_params["family"] = family_name
        if given_name:
            search_params["given"] = given_name
        if birth_date:
            search_params["birthdate"] = birth_date

        try:
            url = urljoin(self.config.fhir_endpoint, "Patient")
            response = self.session.get(
                url, params=search_params, timeout=self.config.timeout_seconds
            )

            if response.status_code == 200:
                bundle = response.json()
                return bundle.get("entry", [])
            else:
                logger.error(f"Patient search failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Patient search error: {e}")
            return []

    def get_patient_encounters(
        self,
        patient_id: str,
        status: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get patient encounters.

        Args:
            patient_id: FHIR Patient resource ID
            status: Encounter status filter
            date_range: Date range tuple (start, end)

        Returns:
            List of encounter resources
        """

        search_params = {"patient": patient_id}

        if status:
            search_params["status"] = status

        if date_range:
            start_date, end_date = date_range
            search_params["date"] = f"ge{start_date.isoformat()}&date=le{end_date.isoformat()}"

        try:
            url = urljoin(self.config.fhir_endpoint, "Encounter")
            response = self.session.get(
                url, params=search_params, timeout=self.config.timeout_seconds
            )

            if response.status_code == 200:
                bundle = response.json()
                return [entry["resource"] for entry in bundle.get("entry", [])]
            else:
                logger.error(f"Encounter search failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Encounter search error: {e}")
            return []

    def _oauth2_authenticate(self) -> bool:
        """Perform OAuth2 authentication."""

        if not self.config.client_id or not self.config.client_secret:
            logger.error("OAuth2 credentials not configured")
            return False

        try:
            # Construct token endpoint (typically /oauth2/token)
            token_url = urljoin(self.config.base_url, "oauth2/token")

            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": "system/Patient.read system/Patient.write system/Observation.write system/CarePlan.write",
            }

            response = requests.post(
                token_url,
                data=auth_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.config.timeout_seconds,
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data["access_token"]

                # Calculate expiration
                expires_in = token_data.get("expires_in", 3600)
                self.token_expires = datetime.now() + timedelta(seconds=expires_in - 60)

                # Add token to session headers
                self.session.headers["Authorization"] = f"Bearer {self.access_token}"

                logger.info("OAuth2 authentication successful")
                return True
            else:
                logger.error(f"OAuth2 authentication failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"OAuth2 authentication error: {e}")
            return False

    def _api_key_authenticate(self) -> bool:
        """Perform API key authentication."""

        if not self.config.api_key:
            logger.error("API key not configured")
            return False

        # Add API key to session headers
        self.session.headers["X-API-Key"] = self.config.api_key
        return True

    def _create_resource(self, resource_type: str, resource_data: Dict[str, Any]) -> Optional[str]:
        """Create a FHIR resource."""

        try:
            url = urljoin(self.config.fhir_endpoint, resource_type)
            response = self.session.post(
                url, json=resource_data, timeout=self.config.timeout_seconds
            )

            if response.status_code in [200, 201]:
                created_resource = response.json()
                resource_id = created_resource.get("id")
                logger.info(f"Created {resource_type} resource: {resource_id}")
                return resource_id
            else:
                logger.error(
                    f"Failed to create {resource_type}: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Resource creation error: {e}")
            return None

    def _refresh_token_if_needed(self):
        """Refresh access token if needed."""

        if self.access_token and self.token_expires and datetime.now() >= self.token_expires:
            logger.info("Access token expired, refreshing...")
            self._oauth2_authenticate()


class HL7Gateway:
    """
    HL7 v2 message processing gateway for legacy system integration.

    Provides bidirectional HL7 v2 message handling for systems that
    don't support modern FHIR interfaces.
    """

    def __init__(self, config: EHRConfiguration):
        """
        Initialize HL7 gateway.

        Args:
            config: EHR system configuration
        """
        self.config = config
        self.status = IntegrationStatus.DISCONNECTED

        # HL7 message delimiters
        self.field_separator = "|"
        self.component_separator = "^"
        self.repetition_separator = "~"
        self.escape_character = "\\"
        self.subcomponent_separator = "&"

        logger.info(f"Initialized HL7 gateway for {config.system_name}")

    def create_oru_message(
        self, patient_mrn: str, risk_assessment: RiskAssessment, encounter_id: Optional[str] = None
    ) -> str:
        """
        Create HL7 ORU^R01 message for suicide risk assessment result.

        Args:
            patient_mrn: Patient medical record number
            risk_assessment: Risk assessment results
            encounter_id: Associated encounter ID

        Returns:
            HL7 message string
        """

        control_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # MSH - Message Header
        msh = self._create_msh_segment(MessageType.ORU_R01, control_id, timestamp)

        # PID - Patient Identification (minimal for result reporting)
        pid = f"PID|1||{patient_mrn}^^^MRN||||||||||||||||||||||||||||||"

        # OBR - Observation Request
        obr = (
            f"OBR|1|{control_id}|{control_id}|72133-2^Suicide risk assessment^LN"
            f"|||{timestamp}||||||||||||||||F|||||||"
        )

        # OBX - Observation Result (Risk Level)
        risk_level_map = {
            RiskLevel.LOW: "LA6112-0^Low^LN",
            RiskLevel.MODERATE: "LA6113-8^Moderate^LN",
            RiskLevel.HIGH: "LA6114-6^High^LN",
            RiskLevel.IMMINENT: "LA6115-3^Critical^LN",
        }

        obx_segments = []

        # Primary risk level result
        obx1 = (
            f"OBX|1|CWE|72133-2^Suicide risk level^LN||"
            f"{risk_level_map[risk_assessment.risk_level]}||||||F|||{timestamp}"
        )
        obx_segments.append(obx1)

        # Risk probability
        obx2 = (
            f"OBX|2|NM|LA6115-3^Risk probability^LN||"
            f"{risk_assessment.risk_probability:.3f}|probability|||||F|||{timestamp}"
        )
        obx_segments.append(obx2)

        # Prediction confidence
        obx3 = (
            f"OBX|3|NM|LA6116-1^Prediction confidence^LN||"
            f"{risk_assessment.prediction_confidence:.3f}|confidence|||||F|||{timestamp}"
        )
        obx_segments.append(obx3)

        # Combine segments
        message_segments = [msh, pid, obr] + obx_segments
        message = "\\r".join(message_segments) + "\\r"

        return message

    def create_adta08_message(
        self, patient_data: Dict[str, Any], risk_assessment: RiskAssessment
    ) -> str:
        """
        Create HL7 ADT^A08 message to update patient information with risk status.

        Args:
            patient_data: Patient demographic data
            risk_assessment: Risk assessment results

        Returns:
            HL7 message string
        """

        control_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # MSH - Message Header
        msh = self._create_msh_segment(MessageType.ADT_A08, control_id, timestamp)

        # PID - Patient Identification
        mrn = patient_data.get("mrn", "")
        last_name = patient_data.get("last_name", "")
        first_name = patient_data.get("first_name", "")
        birth_date = patient_data.get("birth_date", "")
        gender = patient_data.get("gender", "")

        pid = (
            f"PID|1||{mrn}^^^MRN||{last_name}^{first_name}||"
            f"{birth_date}|{gender}||||||||||||||||||||||||||"
        )

        # PV1 - Patient Visit (if encounter info available)
        pv1 = "PV1|1|I||||||||||||||||||||||||||||||||||||||||"

        # AL1 - Patient Allergy Information (use for risk alert)
        # This is a creative use of AL1 to flag high-risk patients
        if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.IMMINENT]:
            al1 = (
                f"AL1|1||SUICIDE_RISK^Suicide Risk Alert^LOCAL||SV^Severe|"
                f"{risk_assessment.risk_level.value.upper()} SUICIDE RISK - See care plan"
            )
        else:
            al1 = ""

        # Combine segments
        message_segments = [msh, pid, pv1]
        if al1:
            message_segments.append(al1)

        message = "\\r".join(message_segments) + "\\r"

        return message

    def parse_hl7_message(self, message: str) -> Dict[str, Any]:
        """
        Parse incoming HL7 message.

        Args:
            message: Raw HL7 message string

        Returns:
            Parsed message dictionary
        """

        try:
            segments = message.replace("\\r", "\\n").split("\\n")
            parsed_message = {"segments": {}}

            for segment in segments:
                if not segment.strip():
                    continue

                fields = segment.split(self.field_separator)
                segment_type = fields[0]

                parsed_message["segments"][segment_type] = {"raw": segment, "fields": fields}

                # Parse specific segment types
                if segment_type == "MSH":
                    parsed_message["message_type"] = fields[8] if len(fields) > 8 else ""
                    parsed_message["control_id"] = fields[9] if len(fields) > 9 else ""
                    parsed_message["timestamp"] = fields[6] if len(fields) > 6 else ""

                elif segment_type == "PID":
                    parsed_message["patient"] = {
                        "mrn": self._extract_identifier(fields[3]) if len(fields) > 3 else "",
                        "name": self._parse_patient_name(fields[5]) if len(fields) > 5 else {},
                        "birth_date": fields[7] if len(fields) > 7 else "",
                        "gender": fields[8] if len(fields) > 8 else "",
                    }

            return parsed_message

        except Exception as e:
            logger.error(f"HL7 message parsing error: {e}")
            return {"error": str(e)}

    def send_hl7_message(self, message: str) -> bool:
        """
        Send HL7 message to configured endpoint.

        Args:
            message: HL7 message string

        Returns:
            True if sent successfully, False otherwise
        """

        if not self.config.hl7_endpoint:
            logger.error("HL7 endpoint not configured")
            return False

        try:
            # This is a simplified HTTP-based HL7 sender
            # In practice, you'd use MLLP (Minimal Lower Layer Protocol)

            response = requests.post(
                self.config.hl7_endpoint,
                data=message,
                headers={"Content-Type": "text/plain", "HL7-Version": "2.5"},
                timeout=self.config.timeout_seconds,
            )

            if response.status_code == 200:
                logger.info("HL7 message sent successfully")
                return True
            else:
                logger.error(f"HL7 message send failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"HL7 message send error: {e}")
            return False

    def _create_msh_segment(
        self, message_type: MessageType, control_id: str, timestamp: str
    ) -> str:
        """Create MSH (Message Header) segment."""

        return (
            f"MSH{self.field_separator}{self.component_separator}"
            f"{self.repetition_separator}{self.escape_character}"
            f"{self.subcomponent_separator}|SUICIDE_CDS|HOSPITAL_SYSTEM|"
            f"EHR_SYSTEM|CLINICAL_SYSTEM|{timestamp}||{message_type.value}|"
            f"{control_id}|P|2.5"
        )

    def _extract_identifier(self, field: str) -> str:
        """Extract identifier from HL7 field."""
        if not field:
            return ""

        components = field.split(self.component_separator)
        return components[0] if components else ""

    def _parse_patient_name(self, field: str) -> Dict[str, str]:
        """Parse patient name field."""
        if not field:
            return {}

        components = field.split(self.component_separator)
        return {
            "family": components[0] if len(components) > 0 else "",
            "given": components[1] if len(components) > 1 else "",
            "middle": components[2] if len(components) > 2 else "",
        }


class ClinicalDecisionSupportEngine:
    """
    Clinical Decision Support (CDS) engine for real-time alerts and recommendations.

    Integrates suicide risk predictions with clinical workflows to provide
    actionable insights at the point of care.
    """

    def __init__(self, fhir_connector: FHIRConnector, hl7_gateway: Optional[HL7Gateway] = None):
        """
        Initialize CDS engine.

        Args:
            fhir_connector: FHIR connector instance
            hl7_gateway: Optional HL7 gateway for legacy systems
        """
        self.fhir_connector = fhir_connector
        self.hl7_gateway = hl7_gateway
        self.active_alerts = {}  # patient_id -> list of active alerts

        logger.info("Initialized Clinical Decision Support engine")

    def process_risk_assessment(
        self,
        patient_id: str,
        risk_assessment: RiskAssessment,
        clinician_id: str,
        encounter_id: Optional[str] = None,
    ) -> List[ClinicalDecisionSupport]:
        """
        Process risk assessment and generate clinical decision support alerts.

        Args:
            patient_id: Patient identifier
            risk_assessment: Risk assessment results
            clinician_id: Current clinician identifier
            encounter_id: Current encounter identifier

        Returns:
            List of clinical decision support alerts
        """

        alerts = []

        # Generate risk-based alerts
        if risk_assessment.risk_level == RiskLevel.IMMINENT:
            alerts.append(
                self._create_imminent_risk_alert(patient_id, clinician_id, risk_assessment)
            )

        elif risk_assessment.risk_level == RiskLevel.HIGH:
            alerts.append(self._create_high_risk_alert(patient_id, clinician_id, risk_assessment))

        elif risk_assessment.risk_level == RiskLevel.MODERATE:
            alerts.append(
                self._create_moderate_risk_alert(patient_id, clinician_id, risk_assessment)
            )

        # Generate safety planning reminder
        if risk_assessment.risk_level != RiskLevel.LOW:
            alerts.append(
                self._create_safety_plan_reminder(patient_id, clinician_id, risk_assessment)
            )

        # Store active alerts
        if patient_id not in self.active_alerts:
            self.active_alerts[patient_id] = []

        self.active_alerts[patient_id].extend(alerts)

        # Send alerts to EHR
        for alert in alerts:
            self._send_alert_to_ehr(alert, patient_id, clinician_id)

        logger.info(f"Generated {len(alerts)} CDS alerts for patient {patient_id}")
        return alerts

    def acknowledge_alert(self, alert_id: str, clinician_id: str) -> bool:
        """
        Acknowledge a clinical alert.

        Args:
            alert_id: Alert identifier
            clinician_id: Clinician acknowledging the alert

        Returns:
            True if acknowledged successfully
        """

        # Find and update alert
        for patient_id, patient_alerts in self.active_alerts.items():
            for alert in patient_alerts:
                if alert.alert_id == alert_id:
                    logger.info(f"Alert {alert_id} acknowledged by {clinician_id}")
                    # In practice, would update EHR task status
                    return True

        return False

    def _create_imminent_risk_alert(
        self, patient_id: str, clinician_id: str, risk_assessment: RiskAssessment
    ) -> ClinicalDecisionSupport:
        """Create imminent risk CDS alert."""

        return ClinicalDecisionSupport(
            alert_id=str(uuid.uuid4()),
            patient_id=patient_id,
            clinician_id=clinician_id,
            alert_type="suicide_risk",
            severity="critical",
            title="IMMINENT SUICIDE RISK ALERT",
            message=(
                f"Patient assessed at IMMINENT suicide risk (confidence: "
                f"{risk_assessment.prediction_confidence:.1%}). IMMEDIATE ACTION REQUIRED."
            ),
            triggers=risk_assessment.acute_risk_factors,
            recommendations=[
                "Implement continuous safety monitoring",
                "Remove all lethal means immediately",
                "Consider involuntary psychiatric hold",
                "Activate crisis response team",
                "Contact psychiatrist on-call immediately",
            ],
            created_timestamp=datetime.now(),
            expiration_timestamp=datetime.now() + timedelta(hours=2),
            actions_available=["acknowledge", "activate_crisis_team", "psychiatric_consult"],
        )

    def _create_high_risk_alert(
        self, patient_id: str, clinician_id: str, risk_assessment: RiskAssessment
    ) -> ClinicalDecisionSupport:
        """Create high risk CDS alert."""

        return ClinicalDecisionSupport(
            alert_id=str(uuid.uuid4()),
            patient_id=patient_id,
            clinician_id=clinician_id,
            alert_type="suicide_risk",
            severity="high",
            title="HIGH SUICIDE RISK ALERT",
            message=(
                f"Patient assessed at HIGH suicide risk (confidence: "
                f"{risk_assessment.prediction_confidence:.1%}). Urgent intervention needed."
            ),
            triggers=risk_assessment.acute_risk_factors,
            recommendations=[
                "Enhanced safety monitoring (q30min checks)",
                "Psychiatric evaluation within 4 hours",
                "Develop comprehensive safety plan",
                "Consider medication adjustment",
                "Increase follow-up frequency",
            ],
            created_timestamp=datetime.now(),
            expiration_timestamp=datetime.now() + timedelta(hours=8),
            actions_available=["acknowledge", "schedule_psychiatric_eval", "create_safety_plan"],
        )

    def _create_moderate_risk_alert(
        self, patient_id: str, clinician_id: str, risk_assessment: RiskAssessment
    ) -> ClinicalDecisionSupport:
        """Create moderate risk CDS alert."""

        return ClinicalDecisionSupport(
            alert_id=str(uuid.uuid4()),
            patient_id=patient_id,
            clinician_id=clinician_id,
            alert_type="suicide_risk",
            severity="medium",
            title="MODERATE SUICIDE RISK",
            message=(
                f"Patient assessed at MODERATE suicide risk (confidence: "
                f"{risk_assessment.prediction_confidence:.1%}). Clinical review recommended."
            ),
            triggers=risk_assessment.acute_risk_factors,
            recommendations=[
                "Clinical review within 24 hours",
                "Update or create safety plan",
                "Consider therapy referral",
                "Schedule follow-up within 1 week",
                "Engage support system",
            ],
            created_timestamp=datetime.now(),
            expiration_timestamp=datetime.now() + timedelta(days=2),
            actions_available=["acknowledge", "schedule_followup", "therapy_referral"],
        )

    def _create_safety_plan_reminder(
        self, patient_id: str, clinician_id: str, risk_assessment: RiskAssessment
    ) -> ClinicalDecisionSupport:
        """Create safety planning reminder alert."""

        return ClinicalDecisionSupport(
            alert_id=str(uuid.uuid4()),
            patient_id=patient_id,
            clinician_id=clinician_id,
            alert_type="clinical_reminder",
            severity="medium",
            title="Safety Plan Recommended",
            message="Patient would benefit from suicide safety plan development or review.",
            triggers=["Elevated suicide risk"],
            recommendations=[
                "Collaborate with patient to create safety plan",
                "Review warning signs and coping strategies",
                "Identify support contacts and resources",
                "Implement means restriction measures",
            ],
            created_timestamp=datetime.now(),
            expiration_timestamp=datetime.now() + timedelta(days=7),
            actions_available=["acknowledge", "create_safety_plan", "review_existing_plan"],
        )

    def _send_alert_to_ehr(
        self, alert: ClinicalDecisionSupport, patient_id: str, clinician_id: str
    ):
        """Send alert to EHR system."""

        try:
            # Send via FHIR Task resource
            if self.fhir_connector.status == IntegrationStatus.CONNECTED:
                clinical_alert = ClinicalAlert(
                    alert_id=alert.alert_id,
                    alert_type=AlertType.PRIMARY_CLINICIAN,
                    urgency_level=(
                        UrgencyLevel.URGENT
                        if alert.severity == "critical"
                        else UrgencyLevel.PRIORITY
                    ),
                    message=alert.message,
                    recipient_role="clinician",
                    patient_identifiers={"id": patient_id},
                    created_timestamp=alert.created_timestamp,
                    recommended_actions=alert.recommendations,
                )

                self.fhir_connector.create_clinical_alert_task(
                    patient_id, clinician_id, clinical_alert
                )

            # Send via HL7 if configured
            if self.hl7_gateway:
                # Could send as HL7 ORM message for order/alert
                pass

        except Exception as e:
            logger.error(f"Failed to send alert to EHR: {e}")


# Integration helper functions
def create_epic_config(base_url: str, client_id: str, client_secret: str) -> EHRConfiguration:
    """Create Epic EHR configuration."""

    return EHRConfiguration(
        system_name="Epic",
        system_type="epic",
        version="2023",
        base_url=base_url,
        fhir_endpoint=urljoin(base_url, "api/FHIR/R4/"),
        auth_type="oauth2",
        client_id=client_id,
        client_secret=client_secret,
        fhir_version=FHIRVersion.R4,
        supported_resources=[
            "Patient",
            "Practitioner",
            "Encounter",
            "Observation",
            "CarePlan",
            "Goal",
            "Task",
            "Communication",
            "DiagnosticReport",
        ],
    )


def create_cerner_config(base_url: str, client_id: str, client_secret: str) -> EHRConfiguration:
    """Create Cerner EHR configuration."""

    return EHRConfiguration(
        system_name="Cerner",
        system_type="cerner",
        version="2023",
        base_url=base_url,
        fhir_endpoint=urljoin(base_url, "v1/"),
        auth_type="oauth2",
        client_id=client_id,
        client_secret=client_secret,
        fhir_version=FHIRVersion.R4,
    )
