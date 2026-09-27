"""
Continuous Scoring Function for Dynamic Priority Queue.
Formula:
Priority Score(t) = W_floor * (6 - ESI_final) + W_risk * P_risk + W_time * ln(1 + t_wait / tau)
"""

import math
from typing import Optional


def compute_priority_score(
    esi_final: int,
    p_risk: float,
    t_wait_mins: float,
    w_floor: float = 1000.0,
    w_risk: float = 100.0,
    w_time_baseline: float = 15.0,
    tau: float = 30.0
) -> float:
    """
    Calculate real-time dynamic priority score for a patient in the ED queue.
    
    Parameters
    ----------
    esi_final : int
        Acuity score (1 = most critical, 5 = least critical).
    p_risk : float
        Continuous risk probability of 30-day mortality or ICU admission (0.0 to 1.0).
    t_wait_mins : float
        Current elapsed waiting time in minutes.
    w_floor : float
        Weight for clinical acuity tier base (default 1000.0).
    w_risk : float
        Weight for intra-tier continuous risk sorting (default 100.0).
    w_time_baseline : float
        Time-decay weight under standard operational load (default 15.0).
    tau : float
        Logarithmic time constant in minutes (default 30.0).
        
    Returns
    -------
    float
        Continuous dynamic priority score (higher score = higher priority in Max Heap).
    """
    # 1. Acuity base component (6 - ESI) * W_floor
    clamped_esi = max(1, min(5, int(esi_final)))
    acuity_base = w_floor * (6.0 - clamped_esi)

    # 2. Intra-tier risk probability component (0.0 - 1.0) * W_risk
    clamped_risk = max(0.0, min(1.0, float(p_risk)))
    intra_tier_risk = w_risk * clamped_risk

    # 3. Sub-linear logarithmic waiting time decay component
    w_time = w_time_baseline
    safe_wait = max(0.0, float(t_wait_mins))
    time_decay = w_time * math.log(1.0 + (safe_wait / tau))

    priority_score = acuity_base + intra_tier_risk + time_decay
    return round(priority_score, 3)

