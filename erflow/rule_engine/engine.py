"""
ESI v5 Safety Floor Rule Engine (ABCDE Safety Aggregator)
Enforces deterministic safety floors:
Final_ESI = min(ML_ESI_Recommendation, ABCDE_ESI_Floor)
Priority is only ever automatically escalated, never downgraded without human clinician override.
"""

from typing import Dict, Any, List, Union
import pandas as pd

from erflow.rule_engine.decision_a import check_decision_point_a
from erflow.rule_engine.decision_b import check_decision_point_b
from erflow.rule_engine.decision_c import estimate_decision_point_c
from erflow.rule_engine.decision_d import check_decision_point_d


class ESIRuleEngine:
    """
    Deterministic ESI v5 Clinical Decision Support Rule Engine.
    """

    def __init__(self):
        pass

    def evaluate(self, patient_data: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
        """
        Evaluate full ESI v5 clinical decision tree for a single patient record.
        
        Evaluation sequence:
        1. Decision Point A: Immediate life threat? -> ESI 1 (Locked)
        2. Decision Point B: High risk situation / altered mental status / severe distress? -> ESI 2
        3. Decision Point D: High-risk vital signs / Geriatric frailty guard? -> ESI 2
        4. Decision Point C: Expected healthcare resources? -> ESI 3, 4, 5
        
        Returns
        -------
        dict
            {
                'acuity_floor': int (1-5),
                'triggered_rules': list of str,
                'is_hard_locked': bool,
                'resource_status': str,
                'decision_details': dict
            }
        """
        if hasattr(patient_data, 'to_dict'):
            p = patient_data.to_dict()
        else:
            p = dict(patient_data)

        triggered_rules: List[str] = []

        # 1. Decision Point A (ESI 1)
        is_esi_1, a_triggers = check_decision_point_a(p)
        if is_esi_1:
            triggered_rules.extend(a_triggers)
            return {
                'acuity_floor': 1,
                'triggered_rules': triggered_rules,
                'is_hard_locked': True,
                'resource_status': 'n/a_immediate_resuscitation',
                'decision_details': {
                    'point_a': a_triggers,
                    'point_b': [],
                    'point_c': 'ESI 1 bypass',
                    'point_d': []
                }
            }

        # 2. Decision Point B (ESI 2)
        is_esi_2_b, b_triggers = check_decision_point_b(p)
        if is_esi_2_b:
            triggered_rules.extend(b_triggers)

        # 3. Decision Point D (ESI 2 - Vital signs danger zone & Frailty Guard)
        is_esi_2_d, d_triggers = check_decision_point_d(p)
        if is_esi_2_d:
            triggered_rules.extend(d_triggers)

        # If either B or D triggered, acuity floor is at least ESI 2
        if is_esi_2_b or is_esi_2_d:
            return {
                'acuity_floor': 2,
                'triggered_rules': triggered_rules,
                'is_hard_locked': True,
                'resource_status': 'bypassed_high_acuity',
                'decision_details': {
                    'point_a': [],
                    'point_b': b_triggers,
                    'point_c': 'ESI 2 bypass',
                    'point_d': d_triggers
                }
            }

        # 4. Decision Point C (Resource Estimation: ESI 3, 4, 5)
        c_floor, resource_status, c_triggers = estimate_decision_point_c(p)
        triggered_rules.extend(c_triggers)

        return {
            'acuity_floor': c_floor,
            'triggered_rules': triggered_rules,
            'is_hard_locked': False,
            'resource_status': resource_status,
            'decision_details': {
                'point_a': [],
                'point_b': [],
                'point_c': c_triggers,
                'point_d': []
            }
        }

    @staticmethod
    def apply_safety_floor(ml_esi_recommendation: int, rule_floor: int) -> int:
        """
        Calculates Final_ESI = min(ML_ESI_Recommendation, ABCDE_ESI_Floor).
        Lower number indicates higher clinical acuity.
        """
        return min(int(ml_esi_recommendation), int(rule_floor))


def evaluate_esi_v5_safety_floor(patient_data: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
    """
    Convenience functional wrapper for evaluating the ESI v5 safety floor.
    """
    engine = ESIRuleEngine()
    return engine.evaluate(patient_data)

