"""
Grounding Validator for Clinical Decision Support.
Enforces structured JSON output and verifies that all clinical claims strictly match
verified SHAP attributions, raw physiological vitals, and deterministic rule engine outputs.
"""

from typing import Dict, Any, List, Tuple


class GroundingValidator:
    """
    Validates clinical explanation claims against empirical model outputs and raw patient data.
    """

    @staticmethod
    def validate_clinical_claims(
        raw_patient_data: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        rule_engine_output: Dict[str, Any],
        generated_summary: Dict[str, Any]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate generated clinical statements against source facts.
        
        Parameters
        ----------
        raw_patient_data : dict
            Ground truth patient intake data.
        shap_explanation : dict
            Verified SHAP outputs from SHAPExplainerService.
        rule_engine_output : dict
            Verified rule engine outputs from ESIRuleEngine.
        generated_summary : dict
            AI/LLM-generated explanation payload to be checked.
            
        Returns
        -------
        (is_valid, violation_reasons, audit_record)
        """
        violations = []
        verified_facts = []

        # 1. Verify ESI Acuity Claim
        # Accept the legacy alias so a narrative from either producer is still
        # checked; the fallback engine used to emit only 'final_esi_recommended',
        # which silently disabled this whole check.
        claimed_esi = generated_summary.get('recommended_esi')
        if claimed_esi is None:
            claimed_esi = generated_summary.get('final_esi_recommended')
        true_floor = rule_engine_output.get('acuity_floor')
        if claimed_esi is not None and true_floor is not None:
            # Claimed acuity cannot be less severe than safety floor
            if int(claimed_esi) > int(true_floor):
                violations.append(
                    f"Acuity Grounding Violation: Claimed ESI {claimed_esi} violates hard deterministic floor ESI {true_floor}"
                )
            else:
                verified_facts.append(f"ESI Level {claimed_esi} respects deterministic floor ESI {true_floor}")

        # 2. Verify Cited Risk Drivers against SHAP values
        top_shap_features = {
            f['feature']: f for f in shap_explanation.get('top_features', [])
        }
        all_shap_features = {
            f['feature']: f for f in shap_explanation.get('all_features_shap', [])
        }

        claimed_drivers = generated_summary.get('primary_risk_factors', [])
        for driver in claimed_drivers:
            feature_name = driver.get('feature')
            claimed_direction = driver.get('direction', 'increasing')

            if feature_name in all_shap_features:
                true_shap = all_shap_features[feature_name]
                is_actually_increasing = true_shap['is_risk_increasing']

                if claimed_direction == 'increasing' and not is_actually_increasing:
                    violations.append(
                        f"SHAP Direction Violation: Feature '{feature_name}' claimed to increase risk, but SHAP value is negative ({true_shap['shap_value']})"
                    )
                elif claimed_direction == 'decreasing' and is_actually_increasing:
                    violations.append(
                        f"SHAP Direction Violation: Feature '{feature_name}' claimed to decrease risk, but SHAP value is positive ({true_shap['shap_value']})"
                    )
                else:
                    verified_facts.append(f"Feature attribution verified: {feature_name} (SHAP={true_shap['shap_value']})")

                # Verify claimed value against raw data
                claimed_val = driver.get('value')
                if claimed_val is not None:
                    actual_val = raw_patient_data.get(feature_name)
                    if actual_val is not None and abs(float(claimed_val) - float(actual_val)) > 0.5:
                        violations.append(
                            f"Numerical Fidelity Violation: '{feature_name}' claimed as {claimed_val}, actual is {actual_val}"
                        )
            else:
                violations.append(
                    f"Hallucination Warning: Claimed risk feature '{feature_name}' is not in demographic model feature space"
                )

        # 3. Verify Deterministic Safety Rules
        claimed_safety_rules = generated_summary.get('triggered_safety_rules', [])
        true_triggered_rules = rule_engine_output.get('triggered_rules', [])

        for rule in claimed_safety_rules:
            # Check for partial match in true triggered rules
            match = any(rule.lower() in t.lower() or t.lower() in rule.lower() for t in true_triggered_rules)
            if not match and true_triggered_rules:
                violations.append(
                    f"Rule Citation Hallucination: Claimed safety rule '{rule}' was not triggered by ESI v5 rules engine"
                )

        is_valid = len(violations) == 0

        audit_record = {
            'is_grounded': is_valid,
            'violations_count': len(violations),
            'violations': violations,
            'verified_facts': verified_facts,
            'acuity_floor_enforced': true_floor,
            'mandatory_review_active': generated_summary.get('requires_human_review', False)
        }

        return is_valid, violations, audit_record

