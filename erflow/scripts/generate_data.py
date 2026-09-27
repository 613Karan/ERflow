"""
Synthetic Emergency Department Dataset Generator
Matches data distributions and clinical characteristics from ERflow_Modelling.ipynb.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_ed_data(n_samples: int = 6230, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic patient data representing realistic Emergency Department cases.
    
    Parameters
    ----------
    n_samples : int
        Number of patient records to synthesize (default 6,230).
    random_seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Synthesized patient records.
    """
    np.random.seed(random_seed)

    patient_ids = [f"PID_{i+1:05d}" for i in range(n_samples)]

    # 1. Demographics & Age Segmentation
    # ~15% pediatric (<18), ~55% adult (18-64), ~30% geriatric (65+)
    age_cohort = np.random.choice(['pediatric', 'adult', 'geriatric'], size=n_samples, p=[0.15, 0.55, 0.30])
    ages = np.empty(n_samples, dtype=int)
    ages[age_cohort == 'pediatric'] = np.random.randint(1, 18, size=np.sum(age_cohort == 'pediatric'))
    ages[age_cohort == 'adult'] = np.random.randint(18, 65, size=np.sum(age_cohort == 'adult'))
    ages[age_cohort == 'geriatric'] = np.random.randint(65, 95, size=np.sum(age_cohort == 'geriatric'))

    genders = np.random.choice(['Female', 'Male'], size=n_samples, p=[0.47, 0.53])

    # Clinical Frailty Scale (CFS: 1 to 9) - baseline median ~2 for younger, higher for geriatrics
    cfs_scores = np.ones(n_samples, dtype=int)
    cfs_scores[age_cohort == 'adult'] = np.random.choice(
        [1, 2, 3, 4], 
        size=np.sum(age_cohort == 'adult'), 
        p=[0.5, 0.3, 0.15, 0.05]
    )
    cfs_scores[age_cohort == 'geriatric'] = np.random.choice(
        range(1, 10), 
        size=np.sum(age_cohort == 'geriatric'),
        p=[0.05, 0.15, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
    )

    # 2. Intake History Availability (50% zero-history / unverified per brief)
    has_prior_history = np.random.choice([1, 0], size=n_samples, p=[0.50, 0.50])
    comorbidity_counts = np.where(
        has_prior_history == 1, 
        np.random.poisson(lam=np.where(age_cohort == 'geriatric', 2.5, 0.8)), 
        0
    )

    # 3. Base Clinical Presentations & ESI-v4 Baseline Ground Truth
    esi_v4 = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.023, 0.271, 0.417, 0.270, 0.019])

    # 4. Generate Age-Calibrated Physiological Vitals based on true acuity
    heart_rate = np.empty(n_samples)
    resp_rate = np.empty(n_samples)
    spo2 = np.empty(n_samples)
    sbp = np.empty(n_samples)
    temp_c = np.empty(n_samples)

    for i in range(n_samples):
        acuity = esi_v4[i]
        cohort = age_cohort[i]
        age = ages[i]

        if cohort == 'pediatric':
            base_hr = 110 if age < 5 else 90
            base_rr = 28 if age < 5 else 20
            base_sbp = 90 + (2 * age)
        elif cohort == 'geriatric':
            base_hr = 72
            base_rr = 18
            base_sbp = 135
        else:  # adult
            base_hr = 75
            base_rr = 16
            base_sbp = 120

        # Vitals distortion based on ESI level
        if acuity == 1:  # Resuscitation / Unstable
            hr = np.random.choice([np.random.normal(145, 15), np.random.normal(40, 8)])
            rr = np.random.choice([np.random.normal(36, 6), np.random.normal(6, 2)])
            sat = np.random.uniform(70, 88)
            bp = np.random.normal(base_sbp - 40, 15)
            temp = np.random.normal(38.8, 1.0)
        elif acuity == 2:  # High Risk / Emergent
            hr = np.random.normal(base_hr + 35, 12)
            rr = np.random.normal(base_rr + 8, 4)
            sat = np.random.uniform(88, 94)
            bp = np.random.normal(base_sbp + 15, 20)
            temp = np.random.normal(38.2, 0.8)
        elif acuity == 3:  # Urgent (Multiple resources, borderline vitals)
            has_abnormal_vital = np.random.rand() < 0.18
            hr = np.random.normal(108, 8) if has_abnormal_vital else np.random.normal(base_hr + 10, 10)
            rr = np.random.normal(24, 3) if has_abnormal_vital else np.random.normal(base_rr + 2, 3)
            sat = np.random.uniform(90, 93) if has_abnormal_vital else np.random.uniform(95, 99)
            bp = np.random.normal(base_sbp, 15)
            temp = np.random.normal(37.4, 0.6)
        elif acuity == 4:  # Less Urgent (1 resource)
            has_abnormal_vital = np.random.rand() < 0.10
            hr = np.random.normal(104, 5) if has_abnormal_vital else np.random.normal(base_hr, 8)
            rr = np.random.normal(22, 2) if has_abnormal_vital else np.random.normal(base_rr, 2)
            sat = np.random.uniform(91, 93) if has_abnormal_vital else np.random.uniform(96, 100)
            bp = np.random.normal(base_sbp, 10)
            temp = np.random.normal(36.8, 0.4)
        else:  # ESI 5 (Non-urgent, 0 resources)
            hr = np.random.normal(base_hr, 6)
            rr = np.random.normal(base_rr, 2)
            sat = np.random.uniform(97, 100)
            bp = np.random.normal(base_sbp, 8)
            temp = np.random.normal(36.6, 0.3)

        heart_rate[i] = max(30.0, min(220.0, hr))
        resp_rate[i] = max(4.0, min(60.0, rr))
        spo2[i] = max(60.0, min(100.0, sat))
        sbp[i] = max(50.0, min(240.0, bp))
        temp_c[i] = max(34.0, min(41.5, temp))

    # 5. Deterministic ABCDE Floor & High-Risk Vital Flags
    hr_flag = np.where(
        age_cohort == 'pediatric', 
        ((ages < 5) & (heart_rate > 140)) | ((ages >= 5) & (heart_rate > 110)), 
        heart_rate > 100
    )
    rr_flag = np.where(
        age_cohort == 'pediatric', 
        ((ages < 5) & (resp_rate > 35)) | ((ages >= 5) & (resp_rate > 24)), 
        resp_rate > 20
    )
    spo2_flag = spo2 < 92.0
    has_high_risk_vitals = (hr_flag | rr_flag | spo2_flag).astype(int)

    # 6. ESI-v5 Simulated Acuity (Automatic upgrade of 3, 4, 5 if high-risk vitals present)
    esi_v5 = esi_v4.copy()
    escalated_mask = (esi_v4 >= 3) & (has_high_risk_vitals == 1)
    esi_v5[escalated_mask] = 2

    # 7. Clinical Outcomes & Complications (Ground Truth Targets)
    icu_prob = np.where(
        esi_v4 == 1, 0.65,
        np.where(esi_v4 == 2, 0.12,
        np.where(esi_v4 == 3, 0.03 + (0.02 * (cfs_scores >= 5)), 0.005))
    )
    admitted_to_icu = (np.random.rand(n_samples) < icu_prob).astype(int)

    mort_prob = np.where(
        esi_v4 == 1, 0.225,
        np.where(esi_v4 == 2, 0.031 + (0.03 * (cfs_scores >= 5)),
        np.where(esi_v4 == 3, 0.016 + (0.02 * (cfs_scores >= 5)), 0.001))
    )
    mortality_30d = (np.random.rand(n_samples) < mort_prob).astype(int)

    # Resource utilization (count of labs, imaging, IVs, consults)
    resources_used = np.where(
        esi_v4 == 1, np.random.choice([4, 5], size=n_samples, p=[0.2, 0.8]),
        np.where(esi_v4 == 2, np.random.choice([2, 3, 4], size=n_samples, p=[0.25, 0.50, 0.25]),
        np.where(esi_v4 == 3, np.random.choice([1, 2, 3], size=n_samples, p=[0.20, 0.50, 0.30]),
        np.where(esi_v4 == 4, 1, 0)))
    )

    # 8. Queue Dynamics & Dynamic Scheduler Features
    base_wait_time_mins = np.random.exponential(scale=np.where(esi_v4 <= 2, 10, 45), size=n_samples)
    is_peak_shift = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    current_wait_time_mins = np.where(is_peak_shift == 1, base_wait_time_mins * 3.0, base_wait_time_mins)

    # Clinician Overrides
    override_occurred = np.where(
        (cfs_scores >= 6) & (esi_v5 >= 3), 1,
        np.where((has_prior_history == 0) & (has_high_risk_vitals == 1) & (esi_v5 >= 3), 1, 0)
    )

    critical_outcome = ((admitted_to_icu == 1) | (mortality_30d == 1)).astype(int)

    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'age_cohort': age_cohort,
        'gender': genders,
        'cfs_frailty_score': cfs_scores,
        'has_prior_history': has_prior_history,
        'comorbidity_count': comorbidity_counts,
        'heart_rate': np.round(heart_rate, 1),
        'resp_rate': np.round(resp_rate, 1),
        'spo2': np.round(spo2, 1),
        'sbp': np.round(sbp, 1),
        'temp_c': np.round(temp_c, 1),
        'has_high_risk_vitals': has_high_risk_vitals,
        'esi_v4_level': esi_v4,
        'esi_v5_level': esi_v5,
        'resources_used': resources_used,
        'current_wait_time_mins': np.round(current_wait_time_mins, 1),
        'is_peak_shift': is_peak_shift,
        'override_occurred': override_occurred,
        'admitted_to_icu': admitted_to_icu,
        'mortality_30d': mortality_30d,
        'critical_outcome': critical_outcome
    })

    return df


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    df = generate_synthetic_ed_data()
    csv_path = os.path.join(data_dir, 'synthetic_ed_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records and saved to {csv_path}")

