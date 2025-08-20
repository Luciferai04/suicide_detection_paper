from __future__ import annotations

from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from suicide_detection.clinical.intervention_recommendations import (
    CarePlanGenerator,
    InterventionEngine,
)
from suicide_detection.clinical.risk_stratification import DSM5Classifier, RiskStratifier
from suicide_detection.models.bert_model import build_model_and_tokenizer

app = FastAPI(title="Suicide Detection Research API", docs_url=None, redoc_url=None)

MODEL_NAME = "mental/mental-bert-base-uncased"
THRESHOLD = 0.5


class InferenceRequest(BaseModel):
    text: str
    max_length: int = 256


class InferenceResponse(BaseModel):
    risk_probability: float
    predicted_label: int


class ClinicalAssessRequest(BaseModel):
    text: str
    max_length: int = 256
    clinical_features: dict | None = None
    patient_context: dict | None = None


class ClinicalAssessResponse(BaseModel):
    risk_probability: float
    risk_level: str
    severity_level: str
    crisis_contact_indicated: bool
    prediction_confidence: float
    high_risk_factors: list
    protective_factors: list


class CarePlanRequest(BaseModel):
    text: str
    patient_context: dict | None = None


class CarePlanResponse(BaseModel):
    care_plan_fhir: dict
    interventions: list


# Lazy-load model/tokenizer and clinical engines on first request
_model = None
_tokenizer = None
_risk = None
_dsm = None
_intervene = None
_careplan = None


def _get_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = build_model_and_tokenizer(
            type("Cfg", (), {"model_name": MODEL_NAME, "num_labels": 2})
        )
        _model.eval()
    return _model, _tokenizer


def _get_clinical_engines():
    global _risk, _dsm, _intervene, _careplan
    if _risk is None:
        _risk = RiskStratifier()
    if _dsm is None:
        _dsm = DSM5Classifier()
    if _intervene is None:
        _intervene = InterventionEngine()
    if _careplan is None:
        _careplan = CarePlanGenerator()
    return _risk, _dsm, _intervene, _careplan


@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest, x_clinician_ok: Optional[str] = Header(default=None)):
    # Safety guard: require clinician oversight header for high-risk results
    model, tokenizer = _get_model()
    enc = tokenizer(
        req.text,
        padding="max_length",
        truncation=True,
        max_length=req.max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items()})
        probs = torch.softmax(out.logits, dim=-1)[0].tolist()
        risk = float(probs[1])
        label = int(risk >= THRESHOLD)
        if label == 1 and not x_clinician_ok:
            raise HTTPException(
                status_code=403,
                detail="High-risk prediction requires clinician authorization header X-Clinician-OK: true",
            )
        return InferenceResponse(risk_probability=risk, predicted_label=label)


@app.post("/clinical/assess", response_model=ClinicalAssessResponse)
async def clinical_assess(
    req: ClinicalAssessRequest, x_clinician_ok: Optional[str] = Header(default=None)
):
    model, tokenizer = _get_model()
    risk_engine, dsm, _, _ = _get_clinical_engines()
    enc = tokenizer(
        req.text,
        padding="max_length",
        truncation=True,
        max_length=req.max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items()})
        probs = torch.softmax(out.logits, dim=-1)[0].tolist()
        risk = float(probs[1])
    # DSM-5 text classification (approximate)
    criteria = dsm.classify_text(req.text, req.clinical_features or {})
    assessment = risk_engine.assess_risk(
        ml_prediction=risk,
        clinical_features=req.clinical_features or {},
        dsm5_criteria=criteria,
        patient_history=req.patient_context or {},
        current_context=req.patient_context or {},
    )
    if assessment.risk_level.name in ("HIGH", "IMMINENT") and not x_clinician_ok:
        raise HTTPException(
            status_code=403,
            detail="High-risk clinical assessment requires clinician authorization header X-Clinician-OK: true",
        )
    return ClinicalAssessResponse(
        risk_probability=assessment.risk_probability,
        risk_level=assessment.risk_level.value,
        severity_level=assessment.severity_level.value,
        crisis_contact_indicated=assessment.crisis_contact_indicated,
        prediction_confidence=assessment.prediction_confidence,
        high_risk_factors=assessment.acute_risk_factors + assessment.chronic_risk_factors,
        protective_factors=assessment.protective_factors,
    )


@app.post("/clinical/careplan", response_model=CarePlanResponse)
async def clinical_careplan(
    req: CarePlanRequest, x_clinician_ok: Optional[str] = Header(default=None)
):
    # Compose assessment -> interventions -> care plan
    model, tokenizer = _get_model()
    risk_engine, dsm, intervention_engine, careplan_gen = _get_clinical_engines()
    enc = tokenizer(
        req.text, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items()})
        probs = torch.softmax(out.logits, dim=-1)[0].tolist()
        risk = float(probs[1])
    criteria = dsm.classify_text(req.text)
    assessment = risk_engine.assess_risk(
        ml_prediction=risk,
        clinical_features={},
        dsm5_criteria=criteria,
        patient_history=req.patient_context or {},
        current_context=req.patient_context or {},
    )
    if assessment.risk_level.name in ("HIGH", "IMMINENT") and not x_clinician_ok:
        raise HTTPException(
            status_code=403,
            detail="High-risk care plan requires clinician authorization header X-Clinician-OK: true",
        )
    interventions = intervention_engine.generate_interventions(
        assessment, req.patient_context or {}
    )
    safety_plan = intervention_engine.create_safety_plan(assessment, req.patient_context or {})
    care_plan = careplan_gen.generate_care_plan(
        assessment, interventions, safety_plan, req.patient_context or {}
    )
    return CarePlanResponse(
        care_plan_fhir=care_plan.__dict__, interventions=[i.__dict__ for i in interventions]
    )
