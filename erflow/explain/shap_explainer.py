"""
SHAP Explainer Service using TreeExplainer on Demographic XGBoost Models.
Extracts top-k feature contributions paired with clinical reference ranges.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import shap


class SHAPExplainerService:
    """
    Computes exact Shapley feature attributions using TreeExplainer.
    """

    def __init__(self, models_dir: Optional[str] = None, data_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

        self.models_dir = models_dir
        self.data_dir = data_dir

        self._load_reference_tables()
        self._load_models()
        self._init_explainers()

    def _load_reference_tables(self):
        ref_path = os.path.join(self.data_dir, 'esi_reference_tables.json')
        if os.path.exists(ref_path):
            with open(ref_path, 'r') as f:
                self.reference_tables = json.load(f)
        else:
            self.reference_tables = {}

    def _load_models(self):
        self.models = {}
        self.feature_names = {
            'geriatric': [
                'heart_rate', 'resp_rate', 'spo2', 'sbp', 'temp_c',
                'cfs_frailty_score', 'has_prior_history', 'comorbidity_count'
            ],
            'adult': [
                'heart_rate', 'resp_rate', 'spo2', 'sbp', 'temp_c',
                'has_prior_history', 'comorbidity_count'
            ],
            'pediatric': [
                'age', 'heart_rate', 'resp_rate', 'spo2', 'sbp', 'temp_c',
                'has_prior_history', 'comorbidity_count'
            ]
        }

        for cohort in ['geriatric', 'adult', 'pediatric']:
            model_file = os.path.join(self.models_dir, f"{cohort}_xgb.json")
            if os.path.exists(model_file):
                booster = xgb.Booster()
                booster.load_model(model_file)
                self.models[cohort] = booster

    def _init_explainers(self):
        self.explainers = {}
        for cohort, booster in self.models.items():
            try:
                self.explainers[cohort] = shap.TreeExplainer(booster)
            except Exception as e:
                print(f"Warning: Could not initialize TreeExplainer for {cohort}: {e}")

    def route_cohort(self, patient: Dict[str, Any]) -> str:
        age = patient.get('age', 35)
        cohort = str(patient.get('age_cohort', '')).lower()
        if cohort in ('pediatric', 'ped'):
            return 'pediatric'
        elif cohort in ('geriatric', 'ger'):
            return 'geriatric'
        elif cohort in ('adult',):
            return 'adult'

        # Fallback to age numbers
        if age < 18:
            return 'pediatric'
        elif age >= 65:
            return 'geriatric'
        else:
            return 'adult'

    def get_reference_range(self, vital_name: str, age: float, cohort: str) -> Dict[str, Any]:
        """
        Lookup clinical normal range and vital threshold for a given vital and age.
        """
        cohort_key = 'adult'
        if cohort == 'pediatric' or age < 18:
            if age < 1:
                cohort_key = 'pediatric_infant'
            elif age < 5:
                cohort_key = 'pediatric_toddler_child'
            else:
                cohort_key = 'pediatric_school_adolescent'
        elif cohort == 'geriatric' or age >= 65:
            cohort_key = 'geriatric'

        age_cohorts = self.reference_tables.get('age_cohorts', {})
        cohort_info = age_cohorts.get(cohort_key, {})
        normal_ranges = cohort_info.get('normal_ranges', {})
        vital_thresholds = cohort_info.get('vital_thresholds', {})

        return {
            'cohort_band': cohort_info.get('name', cohort_key),
            'normal_range': normal_ranges.get(vital_name, None),
            'danger_threshold': vital_thresholds.get(f"{vital_name}_max") or vital_thresholds.get(f"{vital_name}_min")
        }

    def explain_patient(
        self, 
        patient: Dict[str, Any], 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compute SHAP explanation for a patient.
        """
        cohort = self.route_cohort(patient)
        booster = self.models.get(cohort)
        features = self.feature_names.get(cohort, [])

        if booster is None or cohort not in self.explainers:
            return {
                'cohort': cohort,
                'error': f"Model or SHAP explainer for cohort '{cohort}' not available.",
                'top_features': []
            }

        # Build feature vector
        vector = []
        for feat in features:
            val = patient.get(feat)
            if val is None:
                # Fill clinical defaults if absent
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
            vector.append(float(val))

        vector_df = pd.DataFrame([vector], columns=features)
        explainer = self.explainers[cohort]

        shap_values = explainer.shap_values(vector_df)
        if isinstance(shap_values, list):
            # Binary classification raw margin
            shap_arr = np.array(shap_values[-1])[0]
        else:
            shap_arr = np.array(shap_values)[0]

        base_value = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[-1])

        age = float(patient.get('age', 35))
        feature_contributions = []
        for idx, feat_name in enumerate(features):
            val = vector[idx]
            sv = float(shap_arr[idx])
            ref_info = self.get_reference_range(feat_name, age, cohort)

            direction = "INCREASES RISK (Critical)" if sv > 0 else "DECREASES RISK (Protective)"
            feature_contributions.append({
                'feature': feat_name,
                'value': val,
                'shap_value': round(sv, 4),
                'abs_shap': round(abs(sv), 4),
                'direction': direction,
                'is_risk_increasing': sv > 0,
                'clinical_reference': ref_info.get('normal_range'),
                'danger_threshold': ref_info.get('danger_threshold'),
                'cohort_band': ref_info.get('cohort_band')
            })

        # Sort by absolute SHAP impact
        feature_contributions.sort(key=lambda x: x['abs_shap'], reverse=True)
        top_features = feature_contributions[:top_k]

        return {
            'cohort': cohort,
            'features_evaluated': features,
            'base_value': round(base_value, 4),
            'top_features': top_features,
            'all_features_shap': feature_contributions
        }

