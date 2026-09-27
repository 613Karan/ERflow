"""
FastAPI Backend API for ERflow
Provides RESTful endpoints for real-time patient analysis, dynamic Max Heap queue management,
and nurse overrides.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from erflow.agent.orchestrator import AssessmentOrchestrator
from erflow.scheduler.max_heap import DynamicMaxHeapQueue
from erflow.scripts.generate_data import generate_synthetic_ed_data

app = FastAPI(
    title="ERflow REST API",
    description="Emergency Department Clinical Decision Support API with Asymmetric Risk Optimization & Deterministic Safety Floors",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton orchestrator and scheduler queue
orchestrator = AssessmentOrchestrator()
scheduler_queue = DynamicMaxHeapQueue()


# --- Pydantic Schemas ---

class PatientIntakeRequest(BaseModel):
    patient_id: Optional[str] = Field(default=None, description="Patient Identifier")
    age: float = Field(default=45.0, description="Age in years")
    age_cohort: Optional[str] = Field(default=None, description="'pediatric', 'adult', or 'geriatric'")
    gender: Optional[str] = Field(default="Unknown", description="Patient Gender")
    cfs_frailty_score: Optional[int] = Field(default=1, description="Clinical Frailty Scale (1 to 9)")
    has_prior_history: Optional[int] = Field(default=0, description="1 if prior medical records exist, 0 if zero-history")
    comorbidity_count: Optional[int] = Field(default=0, description="Number of known comorbidities")
    heart_rate: Optional[float] = Field(default=80.0, description="Heart rate in beats per minute")
    resp_rate: Optional[float] = Field(default=16.0, description="Respiratory rate in breaths per minute")
    spo2: Optional[float] = Field(default=98.0, description="Oxygen saturation percentage (0-100)")
    sbp: Optional[float] = Field(default=120.0, description="Systolic blood pressure (mmHg)")
    temp_c: Optional[float] = Field(default=37.0, description="Body temperature in Celsius")
    chief_complaint: Optional[str] = Field(default="General ED presentation", description="Presenting complaint")
    requires_lifesaving_intervention: Optional[int] = Field(default=0, description="Decision Point A indicator")
    high_risk_situation: Optional[int] = Field(default=0, description="Decision Point B indicator")
    altered_mental_status: Optional[int] = Field(default=0, description="Decision Point B acute AMS")
    severe_pain_distress: Optional[int] = Field(default=0, description="Decision Point B severe pain/distress")
    acute_chest_pain_ischemic: Optional[int] = Field(default=0, description="Ischemic chest pain / STEMI suspect")
    acute_stroke_symptoms: Optional[int] = Field(default=0, description="Acute stroke / FAST positive symptoms")
    resources_used: Optional[int] = Field(default=None, description="Decision Point C expected resources (None = uncollected)")
    wait_time_mins: Optional[float] = Field(default=0.0, description="Initial wait time in minutes")


class AdvanceTimeRequest(BaseModel):
    delta_minutes: float = Field(default=10.0, ge=0.1, le=120.0, description="Minutes to advance simulated clock")


class NurseOverrideRequest(BaseModel):
    patient_id: str = Field(description="Patient Identifier to override")
    new_esi: int = Field(ge=1, le=5, description="New ESI Acuity level assigned by clinician")
    reason: str = Field(description="Clinical rationale for override")
    clinician_id: str = Field(default="Charge_Nurse_1", description="Staff ID performing override")


class PopulateDemoQueueRequest(BaseModel):
    num_patients: int = Field(default=10, ge=1, le=50, description="Number of demo patients to populate")


# --- Endpoints ---

@app.get("/health")
def health_check():
    """
    Health check and model status.
    """
    return {
        "status": "healthy",
        "service": "ERflow",
        "models_loaded": list(orchestrator.tools.models.keys()),
        "active_queue_size": scheduler_queue.size()
    }


@app.post("/analyze-patient")
def analyze_patient(intake: PatientIntakeRequest):
    """
    Execute full patient assessment without modifying waiting room queue.
    """
    patient_dict = intake.model_dump()
    if not patient_dict.get('patient_id'):
        patient_dict['patient_id'] = f"PID_{abs(hash(str(patient_dict))) % 100000:05d}"

    analysis = orchestrator.analyze_patient(
        patient_data=patient_dict,
        wait_time_mins=float(patient_dict.get('wait_time_mins', 0.0))
    )
    return analysis


@app.get("/queue")
def get_queue_state():
    """
    Get all patients in the ED dynamic priority queue ordered by Max Heap score descending.
    """
    ranked = scheduler_queue.get_ranked_queue()
    return {
        "total_waiting": len(ranked),
        "queue": ranked,
        "override_log_count": len(scheduler_queue.override_audit_log)
    }


@app.post("/queue/add")
def add_patient_to_queue(intake: PatientIntakeRequest):
    """
    Analyze patient and insert into active waiting room dynamic priority queue.
    """
    patient_dict = intake.model_dump()
    if not patient_dict.get('patient_id'):
        patient_dict['patient_id'] = f"PID_{abs(hash(str(patient_dict))) % 100000:05d}"

    analysis = orchestrator.analyze_patient(
        patient_data=patient_dict,
        wait_time_mins=float(patient_dict.get('wait_time_mins', 0.0))
    )

    entry = scheduler_queue.add_patient(
        patient_id=analysis['patient_id'],
        esi_final=analysis['final_esi'],
        p_risk=analysis['p_risk'],
        wait_time_mins=analysis['wait_time_mins'],
        age=int(patient_dict.get('age', 35)),
        age_cohort=analysis['ml_details'].get('cohort', 'adult'),
        gender=patient_dict.get('gender', 'Unknown'),
        chief_complaint=patient_dict.get('chief_complaint', 'General ED presentation'),
        has_prior_history=int(patient_dict.get('has_prior_history', 0)),
        confidence_score=analysis['confidence_score'],
        requires_human_review=analysis['requires_human_review'],
        vital_signs={
            'heart_rate': patient_dict.get('heart_rate'),
            'resp_rate': patient_dict.get('resp_rate'),
            'spo2': patient_dict.get('spo2'),
            'sbp': patient_dict.get('sbp'),
            'temp_c': patient_dict.get('temp_c'),
            'cfs_frailty_score': patient_dict.get('cfs_frailty_score')
        }
    )

    return {
        "message": f"Patient {entry.patient_id} successfully scheduled into dynamic queue.",
        "analysis": analysis,
        "queue_entry": scheduler_queue.get_ranked_queue()
    }


@app.post("/queue/advance-time")
def advance_queue_time(req: AdvanceTimeRequest):
    """
    Advance time across all waiting patients and recalculate dynamic priority scores.
    """
    scheduler_queue.advance_time(req.delta_minutes)
    return {
        "message": f"Advanced waiting time by {req.delta_minutes} minutes.",
        "queue": scheduler_queue.get_ranked_queue()
    }


@app.post("/queue/pop-next")
def pop_next_patient():
    """
    Call highest-priority patient into treatment bed.
    """
    popped = scheduler_queue.pop_next_patient()
    if not popped:
        raise HTTPException(status_code=404, detail="Waiting queue is currently empty.")
    return {
        "message": f"Called patient {popped.patient_id} into ED examination room.",
        "patient": popped,
        "remaining_queue_size": scheduler_queue.size()
    }


@app.post("/override")
def nurse_override_acuity(req: NurseOverrideRequest):
    """
    Record nurse override for an existing patient and re-index heap immediately.
    """
    entry = scheduler_queue.record_nurse_override(
        patient_id=req.patient_id,
        new_esi=req.new_esi,
        reason=req.reason,
        clinician_id=req.clinician_id
    )
    if not entry:
        raise HTTPException(status_code=404, detail=f"Patient {req.patient_id} not found in active queue.")

    return {
        "message": f"Nurse override recorded for patient {req.patient_id}. ESI updated to {req.new_esi}.",
        "updated_patient": entry,
        "queue": scheduler_queue.get_ranked_queue()
    }


@app.post("/queue/populate-demo")
def populate_demo_queue(req: PopulateDemoQueueRequest = PopulateDemoQueueRequest()):
    """
    Populate queue with synthetic ED patients covering pediatric, adult, and geriatric cohorts.
    """
    scheduler_queue.clear()
    df_sample = generate_synthetic_ed_data(n_samples=req.num_patients, random_seed=123)

    added_patients = []
    for _, row in df_sample.iterrows():
        p_data = row.to_dict()
        # Bulk demo population never returns prose; skip the local LLM.
        analysis = orchestrator.analyze_patient(
            patient_data=p_data,
            wait_time_mins=float(p_data.get('current_wait_time_mins', 0.0)),
            use_llm=False
        )
        entry = scheduler_queue.add_patient(
            patient_id=analysis['patient_id'],
            esi_final=analysis['final_esi'],
            p_risk=analysis['p_risk'],
            wait_time_mins=analysis['wait_time_mins'],
            age=int(p_data.get('age', 35)),
            age_cohort=analysis['ml_details'].get('cohort', 'adult'),
            gender=p_data.get('gender', 'Unknown'),
            chief_complaint="Simulated ED Intake",
            has_prior_history=int(p_data.get('has_prior_history', 0)),
            confidence_score=analysis['confidence_score'],
            requires_human_review=analysis['requires_human_review'],
            vital_signs={
                'heart_rate': p_data.get('heart_rate'),
                'resp_rate': p_data.get('resp_rate'),
                'spo2': p_data.get('spo2'),
                'sbp': p_data.get('sbp'),
                'temp_c': p_data.get('temp_c'),
                'cfs_frailty_score': p_data.get('cfs_frailty_score')
            }
        )
        added_patients.append(entry.patient_id)

    return {
        "message": f"Successfully populated queue with {len(added_patients)} patients.",
        "queue": scheduler_queue.get_ranked_queue()
    }

