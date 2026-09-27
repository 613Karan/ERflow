"""
Clinical Decision Support Tool Schemas & Implementations for Agent Orchestration Loop.
Exposed Tools:
1. predict_esi_xgboost(patient_features)
2. get_shap_explanation(patient_features, top_k)
3. run_esi_rule_engine(patient_raw)
4. lookup_reference_range(vital_name, age)
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import xgboost as xgb

from erflow.explain.shap_explainer import SHAPExplainerService
from erflow.explain.uncertainty import compute_tree_variance_confidence
from erflow.rule_engine.engine import ESIRuleEngine


class AssessmentToolsRegistry:
    """
    Unified Tool Execution Registry for Local LLM Agent.
    """

    def __init__(self, models_dir: Optional[str] = None, data_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

        self.models_dir = models_dir
        self.data_dir = data_dir

        self.rule_engine = ESIRuleEngine()
        self.shap_service = SHAPExplainerService(models_dir=models_dir, data_dir=data_dir)
        self._load_models()

    def _load_models(self):
        self.models = {}
        for cohort in ['geriatric', 'adult', 'pediatric']:
            model_path = os.path.join(self.models_dir, f"{cohort}_xgb.json")
            if os.path.exists(model_path):
                booster = xgb.Booster()
                booster.load_model(model_path)
                self.models[cohort] = booster

    def predict_esi_xgboost(self, patient_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the demographic-specific XGBoost model on patient features.
        Returns continuous risk probability P_risk, tree-level confidence, and ML ESI recommendation.
        """
        cohort = self.shap_service.route_cohort(patient_features)
        booster = self.models.get(cohort)
        features_list = self.shap_service.feature_names.get(cohort, [])

        if booster is None:
            return {
                'error': f"Model for cohort '{cohort}' not found in {self.models_dir}",
                'cohort': cohort
            }

        # Build feature vector
        feat_vector = []
        for feat in features_list:
            val = patient_features.get(feat)
            if val is None:
                if feat == 'cfs_frailty_score':
                    val = 2.0
                elif feat == 'has_prior_history':
                    val = 0.0
                elif feat == 'comorbidity_count':
                    val = 0.0
                elif feat == 'age':
                    val = 10.0
                else:
                    val = 0.0
            feat_vector.append(float(val))

        dmatrix = xgb.DMatrix(np.array([feat_vector]), feature_names=features_list)
        raw_margin = float(booster.predict(dmatrix)[0])
        p_risk = 1.0 / (1.0 + np.exp(-raw_margin))

        # Decision threshold (t = 0.504)
        cost_threshold = 0.504
        ml_esi = 2 if p_risk >= cost_threshold else 3

        # Compute tree-level uncertainty
        conf_score, req_review, trajectory, std_dev = compute_tree_variance_confidence(
            model=booster,
            patient_feature_array=feat_vector,
            num_boost_rounds=150,
            step=15,
            mandatory_review_threshold=20.0,
            feature_names=features_list
        )

        return {
            'cohort': cohort,
            'p_risk': round(float(p_risk), 4),
            'cost_sensitive_threshold': cost_threshold,
            'ml_esi_recommendation': ml_esi,
            'ml_action': 'ESCALATE (ESI 2)' if ml_esi == 2 else 'STANDARD (ESI 3+)',
            'confidence_score': conf_score,
            'requires_human_review': req_review,
            'tree_std_dev': std_dev,
            'features_used': features_list
        }

    def get_shap_explanation(self, patient_features: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Extracts SHAP feature attributions paired with clinical normal ranges.
        """
        return self.shap_service.explain_patient(patient_features, top_k=top_k)

    def run_esi_rule_engine(self, patient_raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes deterministic ESI v5 safety floor rule checks.
        """
        return self.rule_engine.evaluate(patient_raw)

    def lookup_reference_range(self, vital_name: str, age: float) -> Dict[str, Any]:
        """
        Look up physiological normal range and critical threshold for a vital sign.
        """
        cohort = 'pediatric' if age < 18 else ('geriatric' if age >= 65 else 'adult')
        return self.shap_service.get_reference_range(vital_name, age=age, cohort=cohort)

    def get_tools_definition(self) -> List[Dict[str, Any]]:
        """
        Return OpenAPI / Ollama tool definition schemas.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "predict_esi_xgboost",
                    "description": "Predict 30-day mortality/ICU risk probability and initial ML ESI using demographic XGBoost agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_features": {
                                "type": "object",
                                "description": "Patient demographic, frailty, and physiological vitals dictionary."
                            }
                        },
                        "required": ["patient_features"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_shap_explanation",
                    "description": "Compute exact SHAP feature attributions for the patient to identify clinical risk drivers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_features": {"type": "object"},
                            "top_k": {"type": "integer", "default": 5}
                        },
                        "required": ["patient_features"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_esi_rule_engine",
                    "description": "Run deterministic ESI v5 ABCDE safety floor checks to prevent normal-vitals danger.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_raw": {"type": "object"}
                        },
                        "required": ["patient_raw"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_reference_range",
                    "description": "Lookup clinical normal range and critical threshold for a vital sign by age.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vital_name": {"type": "string"},
                            "age": {"type": "number"}
                        },
                        "required": ["vital_name", "age"]
                    }
                }
            }
        ]
