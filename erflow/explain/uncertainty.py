"""
Tree-Level Epistemic Uncertainty Quantification
Evaluates prediction consensus across sequential boosting stages of the active XGBoost model.
High variance across iterations indicates conflicting tree signals and model confusion.
"""

from typing import Tuple, List, Union, Optional
import numpy as np
import xgboost as xgb


def compute_tree_variance_confidence(
    model: Union[xgb.Booster, str],
    patient_feature_array: Union[np.ndarray, List[float]],
    num_boost_rounds: int = 150,
    step: int = 15,
    mandatory_review_threshold: float = 20.0,
    feature_names: Optional[List[str]] = None
) -> Tuple[float, bool, List[float], float]:
    """
    Computes Tree-Level Ensemble Variance Score and Confidence Percentage.
    
    Parameters
    ----------
    model : xgb.Booster or str (path to JSON model)
        Trained XGBoost booster.
    patient_feature_array : np.ndarray or list of floats
        Ordered feature vector matching model's expected features.
    num_boost_rounds : int
        Total number of boosting iterations (default 150).
    step : int
        Step size between sub-ranges (default 15).
    mandatory_review_threshold : float
        Confidence score below which mandatory human review is triggered (default 20.0%).
        
    Returns
    -------
    confidence_score : float
        0-100% confidence score.
    requires_human_review : bool
        True if confidence score < mandatory_review_threshold.
    stage_predictions : list of float
        Trajectory of risk probabilities across boosting stages.
    std_dev : float
        Standard deviation across boosting iterations.
    """
    if isinstance(model, str):
        booster = xgb.Booster()
        booster.load_model(model)
    else:
        booster = model

    features_arr = np.asarray(patient_feature_array, dtype=np.float32)
    if features_arr.ndim == 1:
        features_arr = features_arr.reshape(1, -1)

    fn = feature_names if feature_names is not None else getattr(booster, 'feature_names', None)
    if fn is not None:
        d_input = xgb.DMatrix(features_arr, feature_names=fn)
    else:
        d_input = xgb.DMatrix(features_arr)

    # Determine total trees available
    try:
        total_trees = booster.num_boosted_rounds()
    except Exception:
        total_trees = num_boost_rounds

    actual_rounds = min(num_boost_rounds, max(step, total_trees))

    stage_predictions = []
    for k in range(step, actual_rounds + 1, step):
        try:
            raw_margin = booster.predict(d_input, iteration_range=(0, k))[0]
            prob = 1.0 / (1.0 + np.exp(-raw_margin))
            stage_predictions.append(float(prob))
        except Exception:
            # Fallback if iteration_range slicing is constrained
            pass

    if not stage_predictions:
        # Fallback to single full prediction
        raw_margin = booster.predict(d_input)[0]
        prob = 1.0 / (1.0 + np.exp(-raw_margin))
        stage_predictions = [float(prob)]
        std_dev = 0.0
    else:
        std_dev = float(np.std(stage_predictions))

    # Map standard deviation to 0-100% confidence score
    # Normalizing with max standard deviation ~0.25 for Bernoulli probabilities
    normalized_uncertainty = min(1.0, std_dev / 0.25)
    confidence_score = float(max(0.0, min(100.0, (1.0 - normalized_uncertainty) * 100.0)))
    requires_human_review = confidence_score < mandatory_review_threshold

    return round(confidence_score, 1), requires_human_review, stage_predictions, round(std_dev, 4)
