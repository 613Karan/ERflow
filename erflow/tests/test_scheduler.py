"""
Unit tests for Dynamic Priority Scheduler (Continuous Max Heap Queue).
Tests continuous scoring formula, non-inversion across ESI tiers, time advancement,
intra-tier risk sorting, and nurse override re-indexing.
"""

import pytest
from erflow.scheduler.scoring import compute_priority_score
from erflow.scheduler.max_heap import DynamicMaxHeapQueue


def test_priority_score_formula():
    """Verify exact formula calculation."""
    # ESI 3, P_risk=0.50, t_wait=30
    # Acuity base = 1000 * (6 - 3) = 3000
    # Risk base = 100 * 0.50 = 50
    # Time base = 15 * ln(1 + 30/30) = 15 * ln(2) = 15 * 0.693147 = 10.397
    # Expected total = 3060.397
    score = compute_priority_score(esi_final=3, p_risk=0.50, t_wait_mins=30.0)
    assert abs(score - 3060.397) < 0.05


def test_tier_preservation_no_cross_tier_inversion():
    """
    Ensure an ESI 2 patient ALWAYS outranks an ESI 4 patient,
    even if the ESI 4 patient has waited 300 minutes with high risk.
    """
    esi_2_score = compute_priority_score(
        esi_final=2, p_risk=0.05, t_wait_mins=0.0
    )
    esi_4_score_long_wait = compute_priority_score(
        esi_final=4, p_risk=0.99, t_wait_mins=300.0
    )
    
    assert esi_2_score > esi_4_score_long_wait, (
        f"Safety violation: ESI 2 score ({esi_2_score}) must be strictly higher than ESI 4 ({esi_4_score_long_wait})"
    )


def test_intra_tier_risk_sorting():
    """Within the same ESI tier, higher risk probability must rank higher."""
    score_low_risk = compute_priority_score(esi_final=3, p_risk=0.15, t_wait_mins=10.0)
    score_high_risk = compute_priority_score(esi_final=3, p_risk=0.75, t_wait_mins=10.0)
    
    assert score_high_risk > score_low_risk


def test_time_advancement():
    """Test wait time advancement increases score."""
    base_score = compute_priority_score(esi_final=3, p_risk=0.4, t_wait_mins=10.0)
    advanced_score = compute_priority_score(esi_final=3, p_risk=0.4, t_wait_mins=40.0)
    
    assert advanced_score > base_score


def test_dynamic_max_heap_queue_operations():
    """Test dynamic queue addition, ranking, vital deterioration, and clinician override."""
    queue = DynamicMaxHeapQueue()

    # Add 3 patients:
    # P1: ESI 3, Risk 0.2, Wait 10
    # P2: ESI 2, Risk 0.8, Wait 5
    # P3: ESI 4, Risk 0.1, Wait 60
    queue.add_patient("P1", esi_final=3, p_risk=0.2, wait_time_mins=10.0)
    queue.add_patient("P2", esi_final=2, p_risk=0.8, wait_time_mins=5.0)
    queue.add_patient("P3", esi_final=4, p_risk=0.1, wait_time_mins=60.0)

    ranked = queue.get_ranked_queue()
    assert ranked[0]['patient_id'] == "P2"  # ESI 2 must be top
    assert ranked[1]['patient_id'] == "P1"  # ESI 3 second
    assert ranked[2]['patient_id'] == "P3"  # ESI 4 third

    # Advance time by 30 mins
    queue.advance_time(30.0)
    ranked_after_time = queue.get_ranked_queue()
    assert ranked_after_time[0]['patient_id'] == "P2"

    # Deterioration Re-sort: P1 vitals worsen, upgraded to ESI 2 with Risk 0.95
    queue.update_patient_vitals_and_risk("P1", new_p_risk=0.95, new_esi=2)
    ranked_after_deterioration = queue.get_ranked_queue()
    # Now P1 (ESI 2, Risk 0.95, Wait 40) should beat P2 (ESI 2, Risk 0.8, Wait 35)
    assert ranked_after_deterioration[0]['patient_id'] == "P1"

    # Nurse Override: Nurse overrides P3 from ESI 4 to ESI 1 (acute collapse)
    queue.record_nurse_override("P3", new_esi=1, reason="Sudden unresponsive collapse in waiting room")
    ranked_after_override = queue.get_ranked_queue()
    assert ranked_after_override[0]['patient_id'] == "P3"  # P3 is now ESI 1 and top!

    # Pop next patient
    popped = queue.pop_next_patient()
    assert popped.patient_id == "P3"
    assert queue.size() == 2

