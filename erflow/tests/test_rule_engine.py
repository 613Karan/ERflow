"""
Unit tests for Deterministic ESI v5 Safety Floor Rule Engine.
Tests Decision Points A, B, C, D, Geriatric Frailty Guard, and Safety Floor combinations.
"""

import pytest
from erflow.rule_engine.engine import ESIRuleEngine, evaluate_esi_v5_safety_floor
from erflow.rule_engine.decision_a import check_decision_point_a
from erflow.rule_engine.decision_b import check_decision_point_b
from erflow.rule_engine.decision_c import estimate_decision_point_c
from erflow.rule_engine.decision_d import check_decision_point_d


@pytest.fixture
def rule_engine():
    return ESIRuleEngine()


def test_decision_point_a_cardiac_arrest(rule_engine):
    """Test immediate life-threatening cases trigger ESI 1 and lock."""
    patient = {
        'age': 55,
        'requires_lifesaving_intervention': 1,
        'cardiopulmonary_arrest': 1,
        'heart_rate': 0.0,
        'resp_rate': 0.0,
        'spo2': 60.0
    }
    res = rule_engine.evaluate(patient)
    assert res['acuity_floor'] == 1
    assert res['is_hard_locked'] is True
    assert any("Decision Point A" in r for r in res['triggered_rules'])


def test_decision_point_b_acute_stroke(rule_engine):
    """Test acute stroke / altered mental status triggers ESI 2."""
    patient = {
        'age': 68,
        'acute_stroke_symptoms': 1,
        'altered_mental_status': 1,
        'heart_rate': 85.0,
        'resp_rate': 16.0,
        'spo2': 97.0
    }
    res = rule_engine.evaluate(patient)
    assert res['acuity_floor'] == 2
    assert res['is_hard_locked'] is True
    assert any("Decision Point B" in r for r in res['triggered_rules'])


def test_decision_point_d_adult_tachycardia_hypoxia(rule_engine):
    """Test adult vital danger zone triggers ESI 2."""
    patient = {
        'age': 40,
        'age_cohort': 'adult',
        'heart_rate': 115.0,  # > 100
        'resp_rate': 24.0,   # > 20
        'spo2': 90.0         # < 92%
    }
    res = rule_engine.evaluate(patient)
    assert res['acuity_floor'] == 2
    assert any("Decision Point D" in r for r in res['triggered_rules'])


def test_geriatric_frailty_guard(rule_engine):
    """Test elderly patient (65+) with CFS >= 5 and HR > 90 triggers Geriatric Frailty Guard (ESI 2)."""
    patient = {
        'age': 78,
        'age_cohort': 'geriatric',
        'cfs_frailty_score': 6,
        'heart_rate': 94.0,  # > 90 bpm
        'resp_rate': 18.0,
        'spo2': 96.0
    }
    res = rule_engine.evaluate(patient)
    assert res['acuity_floor'] == 2
    assert any("Geriatric" in r for r in res['triggered_rules'])


def test_pediatric_vital_danger_zone(rule_engine):
    """Test pediatric age-banded thresholds (toddler HR > 140 or RR > 35 -> ESI 2)."""
    patient = {
        'age': 3,
        'age_cohort': 'pediatric',
        'heart_rate': 145.0,  # > 140 for toddler
        'resp_rate': 36.0,   # > 35
        'spo2': 95.0
    }
    res = rule_engine.evaluate(patient)
    assert res['acuity_floor'] == 2
    assert any("Decision Point D (Pediatric" in r for r in res['triggered_rules'])


def test_decision_point_c_resources_and_insufficient_data(rule_engine):
    """Test resource estimation mapping and uncollected arrival data fallback."""
    # 2+ resources -> ESI 3
    p3 = {'age': 30, 'resources_used': 3, 'heart_rate': 75, 'resp_rate': 14, 'spo2': 99}
    res3 = rule_engine.evaluate(p3)
    assert res3['acuity_floor'] == 3
    assert res3['resource_status'] == 'sufficient_data'

    # 1 resource -> ESI 4
    p4 = {'age': 30, 'resources_used': 1, 'heart_rate': 75, 'resp_rate': 14, 'spo2': 99}
    res4 = rule_engine.evaluate(p4)
    assert res4['acuity_floor'] == 4

    # 0 resources -> ESI 5
    p5 = {'age': 30, 'resources_used': 0, 'heart_rate': 75, 'resp_rate': 14, 'spo2': 99}
    res5 = rule_engine.evaluate(p5)
    assert res5['acuity_floor'] == 5

    # Missing resources -> stubs as insufficient_data and defaults to standard ESI 3
    p_none = {'age': 30, 'heart_rate': 75, 'resp_rate': 14, 'spo2': 99}
    res_none = rule_engine.evaluate(p_none)
    assert res_none['acuity_floor'] == 3
    assert res_none['resource_status'] == 'insufficient_data'


def test_safety_floor_bounding_rule(rule_engine):
    """Verify Final_ESI = min(ML_ESI, ABCDE_Floor) - AI can only escalate, never downgrade."""
    # If ML predicts ESI 3, but Rule Floor is ESI 2 -> Final must be ESI 2
    assert ESIRuleEngine.apply_safety_floor(ml_esi_recommendation=3, rule_floor=2) == 2

    # If ML predicts ESI 2 (high risk), but Rule Floor is ESI 3 -> Final is ESI 2 (ML escalation)
    assert ESIRuleEngine.apply_safety_floor(ml_esi_recommendation=2, rule_floor=3) == 2

    # If ML predicts ESI 4, but Rule Floor is ESI 1 -> Final is ESI 1 (Rule escalation)
    assert ESIRuleEngine.apply_safety_floor(ml_esi_recommendation=4, rule_floor=1) == 1

