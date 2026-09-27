"""
API integration tests for ERflow FastAPI service.
"""

import pytest
from fastapi.testclient import TestClient
from erflow.api.main import app, scheduler_queue


@pytest.fixture
def client():
    scheduler_queue.clear()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "models_loaded" in data


def test_analyze_patient_endpoint(client):
    payload = {
        "patient_id": "TEST_001",
        "age": 72.0,
        "age_cohort": "geriatric",
        "cfs_frailty_score": 6,
        "heart_rate": 98.0,
        "resp_rate": 20.0,
        "spo2": 95.0,
        "sbp": 130.0,
        "temp_c": 37.2,
        "has_prior_history": 1,
        "comorbidity_count": 2
    }
    response = client.post("/analyze-patient", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "TEST_001"
    assert data["final_esi"] in [1, 2, 3, 4, 5]
    assert "p_risk" in data
    assert "confidence_score" in data
    assert "shap_details" in data
    assert "clinical_narrative" in data


def test_queue_lifecycle_endpoints(client):
    # 1. Add patient
    payload = {
        "patient_id": "PID_QUEUE_1",
        "age": 30.0,
        "heart_rate": 78.0,
        "resp_rate": 16.0,
        "spo2": 99.0,
        "resources_used": 1,
        "wait_time_mins": 5.0
    }
    res_add = client.post("/queue/add", json=payload)
    assert res_add.status_code == 200
    
    # 2. Get queue
    res_q = client.get("/queue")
    assert res_q.status_code == 200
    q_data = res_q.json()
    assert q_data["total_waiting"] == 1
    assert q_data["queue"][0]["patient_id"] == "PID_QUEUE_1"

    # 3. Advance time
    res_adv = client.post("/queue/advance-time", json={"delta_minutes": 15.0})
    assert res_adv.status_code == 200
    q_after_adv = client.get("/queue").json()
    assert q_after_adv["queue"][0]["wait_time_mins"] == 20.0

    # 4. Nurse override
    res_ov = client.post("/override", json={
        "patient_id": "PID_QUEUE_1",
        "new_esi": 2,
        "reason": "Sudden onset chest discomfort",
        "clinician_id": "Nurse_Lead"
    })
    assert res_ov.status_code == 200
    assert res_ov.json()["updated_patient"]["esi_final"] == 2

    # 5. Pop next
    res_pop = client.post("/queue/pop-next")
    assert res_pop.status_code == 200
    assert res_pop.json()["patient"]["patient_id"] == "PID_QUEUE_1"
    assert client.get("/queue").json()["total_waiting"] == 0

