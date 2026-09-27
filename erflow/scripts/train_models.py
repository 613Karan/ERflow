"""
Model Training and Serialization Script for ERflow
Trains 3 demographic-calibrated XGBoost models with custom asymmetric loss functions:
1. Geriatric Agent (65+): alpha = 23.0, includes CFS frailty score
2. Adult Agent (18-64): alpha = 18.0
3. Pediatric Agent (<18): alpha = 28.0, includes age-stratified baseline features

Exports models as JSON to models/{geriatric,adult,pediatric}_xgb.json.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

from erflow.scripts.generate_data import generate_synthetic_ed_data


def create_asymmetric_objective(alpha: float, beta: float = 1.0):
    """
    Creates custom asymmetric logistic loss objective.
    Heavily penalizes False Negatives (critical cases classified as standard) by factor alpha.
    """
    def asymmetric_logistic_obj(preds, dmatrix):
        labels = dmatrix.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        grad = p * (alpha * labels + beta * (1.0 - labels)) - alpha * labels
        hess = p * (1.0 - p) * (alpha * labels + beta * (1.0 - labels))
        return grad, hess
    return asymmetric_logistic_obj


def train_and_export_all_models(models_dir: str = None, data_df: pd.DataFrame = None) -> dict:
    """
    Train and export the trio of demographic models.
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    if data_df is None:
        data_df = generate_synthetic_ed_data()

    results = {}

    # Common XGBoost parameters
    base_params = {
        'max_depth': 4,
        'learning_rate': 0.01,
        'tree_method': 'hist',
        'min_child_weight': 1,
        'disable_default_eval_metric': 1
    }

    # ==========================================
    # 1. GERIATRIC AGENT (65+)
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING GERIATRIC AGENT (Ages 65+, alpha=23.0)")
    print("="*50)
    df_geriatric = data_df[data_df['age_cohort'] == 'geriatric'].copy()
    features_geriatric = [
        'heart_rate', 'resp_rate', 'spo2', 'sbp', 'temp_c',
        'cfs_frailty_score', 'has_prior_history', 'comorbidity_count'
    ]
    X_geri = df_geriatric[features_geriatric]
    y_geri = df_geriatric['critical_outcome']

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X_geri, y_geri, test_size=0.2, stratify=y_geri, random_state=42
    )
    dtrain_g = xgb.DMatrix(X_train_g, label=y_train_g)
    dtest_g = xgb.DMatrix(X_test_g, label=y_test_g)

    obj_geri = create_asymmetric_objective(alpha=23.0, beta=1.0)
    geriatric_agent = xgb.train(
        base_params,
        dtrain_g,
        num_boost_round=150,
        obj=obj_geri
    )

    geri_model_path = os.path.join(models_dir, 'geriatric_xgb.json')
    geriatric_agent.save_model(geri_model_path)

    raw_g = geriatric_agent.predict(dtest_g)
    probs_g = 1.0 / (1.0 + np.exp(-raw_g))
    y_pred_g = (probs_g >= 0.504).astype(int)
    auc_g = roc_auc_score(y_test_g, probs_g)
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_test_g, y_pred_g).ravel()

    results['geriatric'] = {
        'model_path': geri_model_path,
        'features': features_geriatric,
        'alpha': 23.0,
        'threshold': 0.504,
        'roc_auc': float(auc_g),
        'tp': int(tp_g),
        'tn': int(tn_g),
        'fp': int(fp_g),
        'fn': int(fn_g),
        'sensitivity': float(tp_g / (tp_g + fn_g)) if (tp_g + fn_g) > 0 else 0.0,
        'specificity': float(tn_g / (tn_g + fp_g)) if (tn_g + fp_g) > 0 else 0.0
    }
    print(f"Geriatric ROC-AUC: {auc_g:.3f}, Sensitivity (Catch Rate): {results['geriatric']['sensitivity']*100:.1f}%, Saved to {geri_model_path}")

    # ==========================================
    # 2. ADULT AGENT (18-64)
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING ADULT AGENT (Ages 18-64, alpha=18.0)")
    print("="*50)
    df_adult = data_df[data_df['age_cohort'] == 'adult'].copy()
    features_adult = [
        'heart_rate', 'resp_rate', 'spo2', 'sbp', 'temp_c',
        'has_prior_history', 'comorbidity_count'
    ]
    X_adult = df_adult[features_adult]
    y_adult = df_adult['critical_outcome']

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_adult, y_adult, test_size=0.2, stratify=y_adult, random_state=42
    )
    dtrain_a = xgb.DMatrix(X_train_a, label=y_train_a)
    dtest_a = xgb.DMatrix(X_test_a, label=y_test_a)

    obj_adult = create_asymmetric_objective(alpha=18.0, beta=1.0)
    adult_agent = xgb.train(
        base_params,
        dtrain_a,
        num_boost_round=150,
        obj=obj_adult
    )

    adult_model_path = os.path.join(models_dir, 'adult_xgb.json')
    adult_agent.save_model(adult_model_path)

    raw_a = adult_agent.predict(dtest_a)
    probs_a = 1.0 / (1.0 + np.exp(-raw_a))
    y_pred_a = (probs_a >= 0.504).astype(int)
    auc_a = roc_auc_score(y_test_a, probs_a)
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test_a, y_pred_a).ravel()

    results['adult'] = {
        'model_path': adult_model_path,
        'features': features_adult,
        'alpha': 18.0,
        'threshold': 0.504,
        'roc_auc': float(auc_a),
        'tp': int(tp_a),
        'tn': int(tn_a),
        'fp': int(fp_a),
        'fn': int(fn_a),
        'sensitivity': float(tp_a / (tp_a + fn_a)) if (tp_a + fn_a) > 0 else 0.0,
        'specificity': float(tn_a / (tn_a + fp_a)) if (tn_a + fp_a) > 0 else 0.0
    }
    print(f"Adult ROC-AUC: {auc_a:.3f}, Sensitivity (Catch Rate): {results['adult']['sensitivity']*100:.1f}%, Saved to {adult_model_path}")

    # ==========================================
    # 3. PEDIATRIC AGENT (<18)
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING PEDIATRIC AGENT (Ages <18, alpha=28.0)")
    print("="*50)
    df_pediatric = data_df[data_df['age_cohort'] == 'pediatric'].copy()
    features_pediatric = [
        'age', 'heart_rate', 'resp_rate', 'spo2', 'sbp', 'temp_c',
        'has_prior_history', 'comorbidity_count'
    ]
    X_ped = df_pediatric[features_pediatric]
    y_ped = df_pediatric['critical_outcome']

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_ped, y_ped, test_size=0.2, stratify=y_ped, random_state=42
    )
    dtrain_p = xgb.DMatrix(X_train_p, label=y_train_p)
    dtest_p = xgb.DMatrix(X_test_p, label=y_test_p)

    obj_ped = create_asymmetric_objective(alpha=28.0, beta=1.0)
    pediatric_agent = xgb.train(
        base_params,
        dtrain_p,
        num_boost_round=150,
        obj=obj_ped
    )

    ped_model_path = os.path.join(models_dir, 'pediatric_xgb.json')
    pediatric_agent.save_model(ped_model_path)

    raw_p = pediatric_agent.predict(dtest_p)
    probs_p = 1.0 / (1.0 + np.exp(-raw_p))
    y_pred_p = (probs_p >= 0.504).astype(int)
    auc_p = roc_auc_score(y_test_p, probs_p)
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_test_p, y_pred_p).ravel()

    results['pediatric'] = {
        'model_path': ped_model_path,
        'features': features_pediatric,
        'alpha': 28.0,
        'threshold': 0.504,
        'roc_auc': float(auc_p),
        'tp': int(tp_p),
        'tn': int(tn_p),
        'fp': int(fp_p),
        'fn': int(fn_p),
        'sensitivity': float(tp_p / (tp_p + fn_p)) if (tp_p + fn_p) > 0 else 0.0,
        'specificity': float(tn_p / (tn_p + fp_p)) if (tn_p + fp_p) > 0 else 0.0
    }
    print(f"Pediatric ROC-AUC: {auc_p:.3f}, Sensitivity (Catch Rate): {results['pediatric']['sensitivity']*100:.1f}%, Saved to {ped_model_path}")

    # Save summary metadata
    metadata_path = os.path.join(models_dir, 'models_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved models metadata to {metadata_path}")

    return results


if __name__ == '__main__':
    train_and_export_all_models()

