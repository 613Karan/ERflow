"""
Decision Point D: Danger Zone Vital Signs Assessment & Geriatric Frailty Guard (ESI Level 2)
Evaluates physiological vital sign deviations across pediatric, adult, and geriatric populations.
"""

from typing import Dict, Any, Tuple, List


def check_decision_point_d(patient: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if patient triggers Decision Point D danger zone vitals or Geriatric Frailty Guard.
    
    Age-stratified criteria:
    - Pediatric (<1 yr): HR > 160, RR > 45, SpO2 < 92%
    - Pediatric (1-4 yrs): HR > 140, RR > 35, SpO2 < 92%
    - Pediatric (5-17 yrs): HR > 110, RR > 24, SpO2 < 92%
    - Adult (18-64 yrs): HR > 100, RR > 20, SpO2 < 92%
    - Geriatric (65+ yrs): HR > 100, RR > 20, SpO2 < 92%
    - Geriatric Frailty Guard: Age >= 65, CFS >= 5, HR > 90 bpm
    
    Returns
    -------
    (is_danger_zone, triggered_reasons)
    """
    triggers = []
    
    age = patient.get('age', 35)
    cohort = patient.get('age_cohort', 'adult')
    if isinstance(cohort, str):
        cohort = cohort.lower()

    is_pediatric = (cohort == 'pediatric') or (age < 18)
    is_geriatric = (cohort == 'geriatric') or (age >= 65)

    hr = patient.get('heart_rate')
    rr = patient.get('resp_rate')
    spo2 = patient.get('spo2')
    sbp = patient.get('sbp')
    temp_c = patient.get('temp_c')
    cfs = patient.get('cfs_frailty_score', 1)

    # 1. Pediatric Age-Banded Vital Thresholds
    if is_pediatric:
        if age < 1:
            hr_limit = 160
            rr_limit = 45
            age_label = "Infant (<1 yr)"
        elif age < 5:
            hr_limit = 140
            rr_limit = 35
            age_label = "Toddler/Young Child (1-4 yrs)"
        else:
            hr_limit = 110
            rr_limit = 24
            age_label = "Child/Adolescent (5-17 yrs)"

        if hr is not None and hr > hr_limit:
            triggers.append(f"Decision Point D (Pediatric {age_label}): Tachycardia (HR {hr} > {hr_limit} bpm)")
        if rr is not None and rr > rr_limit:
            triggers.append(f"Decision Point D (Pediatric {age_label}): Tachypnea (RR {rr} > {rr_limit}/min)")
        if spo2 is not None and spo2 < 92.0:
            triggers.append(f"Decision Point D (Pediatric): Hypoxia (SpO2 {spo2}% < 92%)")
        if temp_c is not None and age < 0.25 and temp_c >= 38.0:  # Neonatal fever
            triggers.append(f"Decision Point D (Pediatric Neonate): High-risk fever in infant < 3 months (Temp {temp_c}°C)")

    # 2. Adult & Geriatric Vital Thresholds
    else:
        if hr is not None and hr > 100:
            triggers.append(f"Decision Point D (Adult/Geriatric): Tachycardia (HR {hr} > 100 bpm)")
        if rr is not None and rr > 20:
            triggers.append(f"Decision Point D (Adult/Geriatric): Tachypnea (RR {rr} > 20/min)")
        if spo2 is not None and spo2 < 92.0:
            triggers.append(f"Decision Point D (Adult/Geriatric): Hypoxemia (SpO2 {spo2}% < 92%)")
        if sbp is not None and sbp < 90:
            triggers.append(f"Decision Point D (Adult/Geriatric): Hypotension (SBP {sbp} < 90 mmHg)")

    # 3. Geriatric Frailty Guard (Diminished physiological reserve)
    if is_geriatric and cfs is not None and cfs >= 5:
        if hr is not None and hr > 90:
            triggers.append(f"Geriatric Frailty Guard: Vulnerable baseline (CFS={cfs} >= 5) with elevated heart rate (HR {hr} > 90 bpm)")

    is_danger_zone = len(triggers) > 0
    return is_danger_zone, triggers

