"""
Unit tests for Grounding Validator and Hallucination Rejection.
Tests mathematical verification of clinical narrative claims against empirical SHAP values,
raw patient vitals, and deterministic ESI v5 safety floors.
"""

import pytest
from erflow.explain.grounding import GroundingValidator


@pytest.fixture
def sample_grounding_context():
    raw_patient = {
        'age': 50,
        'heart_rate': 110.0,
        'resp_rate': 18.0,
        'spo2': 98.0,
        'sbp': 130.0,
        'cfs_frailty_score': 2
    }
    
    shap_explanation = {
        'top_features': [
            {'feature': 'heart_rate', 'value': 110.0, 'shap_value': 0.85, 'is_risk_increasing': True},
            {'feature': 'spo2', 'value': 98.0, 'shap_value': -0.42, 'is_risk_increasing': False}
        ],
        'all_features_shap': [
            {'feature': 'heart_rate', 'value': 110.0, 'shap_value': 0.85, 'is_risk_increasing': True},
            {'feature': 'resp_rate', 'value': 18.0, 'shap_value': -0.10, 'is_risk_increasing': False},
            {'feature': 'spo2', 'value': 98.0, 'shap_value': -0.42, 'is_risk_increasing': False}
        ]
    }
    
    rule_output = {
        'acuity_floor': 2,
        'triggered_rules': ["Decision Point D (Adult/Geriatric): Tachycardia (HR 110 > 100 bpm)"],
        'is_hard_locked': True
    }
    
    return raw_patient, shap_explanation, rule_output


def test_valid_clinical_claims_pass_grounding(sample_grounding_context):
    """Test correctly formulated grounded clinical summary passes validation."""
    raw, shap, rules = sample_grounding_context
    
    valid_summary = {
        'recommended_esi': 2,
        'primary_risk_factors': [
            {'feature': 'heart_rate', 'value': 110.0, 'direction': 'increasing'}
        ],
        'triggered_safety_rules': [
            "Decision Point D (Adult/Geriatric): Tachycardia"
        ]
    }
    
    is_valid, violations, audit = GroundingValidator.validate_clinical_claims(
        raw, shap, rules, valid_summary
    )
    
    assert is_valid is True
    assert len(violations) == 0
    assert audit['is_grounded'] is True


def test_hallucination_of_protective_feature_as_risk_is_rejected(sample_grounding_context):
    """Test hallucination claiming normal/protective SpO2 increased risk is caught."""
    raw, shap, rules = sample_grounding_context
    
    hallucinatory_summary = {
        'recommended_esi': 2,
        'primary_risk_factors': [
            {'feature': 'spo2', 'value': 98.0, 'direction': 'increasing'}  # False: SpO2 98% has negative SHAP!
        ]
    }
    
    is_valid, violations, audit = GroundingValidator.validate_clinical_claims(
        raw, shap, rules, hallucinatory_summary
    )
    
    assert is_valid is False
    assert any("SHAP Direction Violation" in v for v in violations)


def test_acuity_floor_violation_is_rejected(sample_grounding_context):
    """Test claim that assigns ESI 4 when safety floor is ESI 2 is caught."""
    raw, shap, rules = sample_grounding_context
    
    invalid_floor_summary = {
        'recommended_esi': 4,  # Violates floor ESI 2!
        'primary_risk_factors': []
    }
    
    is_valid, violations, audit = GroundingValidator.validate_clinical_claims(
        raw, shap, rules, invalid_floor_summary
    )
    
    assert is_valid is False
    assert any("Acuity Grounding Violation" in v for v in violations)


def test_numerical_mismatch_is_rejected(sample_grounding_context):
    """Test claiming heart rate was 140 when raw record was 110 is caught."""
    raw, shap, rules = sample_grounding_context
    
    mismatched_summary = {
        'recommended_esi': 2,
        'primary_risk_factors': [
            {'feature': 'heart_rate', 'value': 140.0, 'direction': 'increasing'}  # Actual is 110
        ]
    }
    
    is_valid, violations, audit = GroundingValidator.validate_clinical_claims(
        raw, shap, rules, mismatched_summary
    )
    
    assert is_valid is False
    assert any("Numerical Fidelity Violation" in v for v in violations)



# ---------------------------------------------------------------------------
# Producer/validator contract
#
# The checks above validate the validator in isolation, using hand-written
# dictionaries. They passed while the acuity-floor check was dead code, because
# the fallback engine emitted 'final_esi_recommended' and the validator read
# 'recommended_esi'. These tests feed real generate_clinical_trace() output into
# the validator so the two halves cannot drift apart again.
# ---------------------------------------------------------------------------

from erflow.agent.llm_client import OllamaLLMClient, REQUIRED_NARRATIVE_KEYS


@pytest.fixture
def deterministic_trace_inputs():
    patient = {'age': 76, 'heart_rate': 122.0, 'resp_rate': 28.0, 'spo2': 88.0, 'sbp': 86.0}
    ml_result = {
        'cohort': 'geriatric',
        'p_risk': 0.89,
        'ml_esi_recommendation': 2,
        'confidence_score': 71.5,
        'requires_human_review': False,
    }
    rule_result = {
        'acuity_floor': 2,
        'triggered_rules': ['Decision Point D (Adult/Geriatric): Hypoxemia (SpO2 88.0% < 92%)'],
    }
    shap_feature = {
        'feature': 'spo2',
        'value': 88.0,
        'shap_value': 1.2432,
        'abs_shap': 1.2432,
        'is_risk_increasing': True,
        'clinical_reference': [94, 100],
    }
    shap_result = {'top_features': [shap_feature], 'all_features_shap': [shap_feature]}
    return patient, ml_result, rule_result, shap_result


def test_deterministic_narrative_exposes_every_required_key(deterministic_trace_inputs):
    """The fallback engine must satisfy the contract the rest of the system reads."""
    patient, ml_result, rule_result, shap_result = deterministic_trace_inputs
    trace = OllamaLLMClient._generate_deterministic_clinical_trace(
        None, patient, ml_result, rule_result, shap_result, final_esi=2
    )
    narrative = trace['structured_narrative']

    for key in REQUIRED_NARRATIVE_KEYS:
        assert key in narrative, f"fallback narrative is missing '{key}'"
    assert trace['provider'] == 'deterministic_clinical_engine'


def test_acuity_floor_check_actually_runs_on_real_narrative(deterministic_trace_inputs):
    """
    Regression guard for the dead acuity-floor check: a real narrative must
    produce a verified ESI fact, not silently skip the comparison.
    """
    patient, ml_result, rule_result, shap_result = deterministic_trace_inputs
    trace = OllamaLLMClient._generate_deterministic_clinical_trace(
        None, patient, ml_result, rule_result, shap_result, final_esi=2
    )

    is_valid, violations, audit = GroundingValidator.validate_clinical_claims(
        patient, shap_result, rule_result, trace['structured_narrative']
    )

    assert is_valid is True, violations
    assert any("respects deterministic floor" in f for f in audit['verified_facts']), \
        "acuity floor check did not execute against the real narrative"


def test_unsafe_claim_in_real_narrative_shape_is_rejected(deterministic_trace_inputs):
    """Downgrading acuity below the safety floor must be caught on the real key."""
    patient, ml_result, rule_result, shap_result = deterministic_trace_inputs
    trace = OllamaLLMClient._generate_deterministic_clinical_trace(
        None, patient, ml_result, rule_result, shap_result, final_esi=2
    )
    narrative = trace['structured_narrative']
    # Claim ESI 4 against a floor of ESI 2.
    narrative['recommended_esi'] = 4
    narrative['final_esi_recommended'] = 4

    is_valid, violations, audit = GroundingValidator.validate_clinical_claims(
        patient, shap_result, rule_result, narrative
    )

    assert is_valid is False
    assert any("Acuity Grounding Violation" in v for v in violations)


def test_legacy_alias_still_reaches_the_floor_check(deterministic_trace_inputs):
    """A narrative carrying only the old key name must still be validated."""
    patient, ml_result, rule_result, shap_result = deterministic_trace_inputs
    legacy_narrative = {
        'final_esi_recommended': 4,  # Violates floor ESI 2
        'primary_risk_factors': [],
        'triggered_safety_rules': [],
    }

    is_valid, violations, audit = GroundingValidator.validate_clinical_claims(
        patient, shap_result, rule_result, legacy_narrative
    )

    assert is_valid is False
    assert any("Acuity Grounding Violation" in v for v in violations)


def test_ollama_host_env_var_is_honoured_by_the_probe(monkeypatch):
    """The readiness probe must target the configured host, not a hardcoded one."""
    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.1:9999")
    client = OllamaLLMClient()

    assert client._host == "192.0.2.1"
    assert client._port == 9999
    # RFC 5737 documentation address: unreachable, so the fallback must engage.
    assert client.is_connected is False


def test_llm_output_missing_contract_keys_is_rejected():
    """Partial model output must be refused rather than shown half-empty."""
    client = OllamaLLMClient.__new__(OllamaLLMClient)
    narrative, reason = client._coerce_llm_narrative(
        {'clinical_rationale': 'looks fine but incomplete'},
        final_esi=2,
        ml_result={'p_risk': 0.5},
    )

    assert narrative is None
    assert "missing required keys" in reason


def test_llm_output_accepts_legacy_esi_key():
    """A model answering with the legacy key name is normalised, not discarded."""
    client = OllamaLLMClient.__new__(OllamaLLMClient)
    narrative, reason = client._coerce_llm_narrative(
        {
            'final_esi_recommended': 2,
            'clinical_rationale': 'Hypoxemia drives the escalation.',
            'primary_risk_factors': [],
            'triggered_safety_rules': [],
        },
        final_esi=2,
        ml_result={'p_risk': 0.89, 'confidence_score': 71.5},
    )

    assert reason is None
    assert narrative['recommended_esi'] == 2
