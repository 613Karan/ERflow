# ERflow Dataset

**File:** [`erflow/data/erflow_ed_dataset.parquet`](erflow/data/erflow_ed_dataset.parquet)
**Rows:** 6,230 (one row per simulated Emergency Department patient) · **Columns:** 24 · **Size:** ~95 KB (zstd)
**Origin:** 100% synthetic. Every row has `data_origin = "synthetic"`. No real patient data is used anywhere in ERflow.

This is the exact dataset the three XGBoost risk models were trained and tested on. The `model_split` column shows which rows trained each model and which were held out for the metrics in the [README](README.md#2-model-training).

---

## 1. Loading it

```python
import pandas as pd
df = pd.read_parquet("erflow/data/erflow_ed_dataset.parquet")
```

Provenance is also stored inside the file as Parquet key-value metadata (generator, seed, row count, export time):

```python
import pyarrow.parquet as pq
pq.read_schema("erflow/data/erflow_ed_dataset.parquet").metadata
```

## 2. Reproducing it

The data is fully deterministic. Running the command below regenerates identical rows every time. Only the `erflow.exported_at_utc` metadata field changes:

```bash
python -m erflow.scripts.export_parquet
```

| Setting | Value |
|---|---|
| Generator | `generate_synthetic_ed_data()` in [`erflow/scripts/generate_data.py`](erflow/scripts/generate_data.py) (prototyped in `ERflow_Modelling.ipynb`) |
| Random seed | `42` (NumPy legacy `RandomState`, stable across NumPy versions) |
| Rows | `6230` |
| Train/test split | Per age cohort, stratified on `critical_outcome`, 80/20, `random_state=42`. Same as [`train_models.py`](erflow/scripts/train_models.py) |

## 3. Summary

| Cohort | Rows | Train | Test | Critical outcomes | Rate |
|---|---:|---:|---:|---:|---:|
| Pediatric (<18) | 970 | 776 | 194 | 80 | 8.2% |
| Adult (18–64) | 3,431 | 2,744 | 687 | 298 | 8.7% |
| Geriatric (65+) | 1,829 | 1,463 | 366 | 158 | 8.6% |
| **Total** | **6,230** | **4,983** | **1,247** | **536** | **8.6%** |

- **Gender:** 3,279 male / 2,951 female
- **Prior history available:** 49.6% (the other half are "zero-history" arrivals)
- **ICU admissions:** 417 · **30-day deaths:** 145 · **Either (critical outcome):** 536
- **Critical-outcome rate by baseline ESI:** ESI 1: 74.4% · ESI 2: 15.8% · ESI 3: 5.8% · ESI 4: 0.5% · ESI 5: 0.8%

## 4. Schema

**Kind** says how each value is produced:
- **sampled**: drawn at random from a distribution
- **derived**: computed from other columns with a fixed rule
- **label**: an outcome used as a training target
- **meta**: bookkeeping

| Column | Type | Unit / values | Kind | Meaning and how it is produced |
|---|---|---|---|---|
| `patient_id` | string | `PID_00001` … `PID_06230` | meta | Sequential identifier. Not linked to any person. |
| `age` | int16 | years, 1–94 | sampled | Uniform within the cohort: 1–17, 18–64 or 65–94. |
| `age_cohort` | string | `pediatric` / `adult` / `geriatric` | sampled | Drawn with p = 0.15 / 0.55 / 0.30. Decides which of the three models a patient is scored by. |
| `gender` | string | `Female` / `Male` | sampled | p = 0.47 / 0.53. Not used by any model. |
| `cfs_frailty_score` | int8 | Clinical Frailty Scale, 1–9 | sampled | Always 1 for pediatric. Adult: 1–4 skewed low. Geriatric: 1–9 peaking at 3. Geriatric model feature. |
| `has_prior_history` | int8 | 0 / 1 | sampled | 1 if a verified medical history is available (p = 0.5). |
| `comorbidity_count` | int8 | count, 0–10 | sampled | Poisson (λ = 2.5 geriatric, 0.8 otherwise) when history is available, otherwise 0. |
| `heart_rate` | float64 | beats/min, 30–220 | sampled | Normal around an age-appropriate baseline, shifted by `esi_v4_level` (see §5). Clipped to range. |
| `resp_rate` | float64 | breaths/min, 4–60 | sampled | Same approach as heart rate. |
| `spo2` | float64 | %, 60–100 | sampled | Uniform band set by `esi_v4_level`: 70–88% for ESI 1, up to 97–100% for ESI 5. |
| `sbp` | float64 | mmHg, 50–240 | sampled | Systolic blood pressure around an age baseline (pediatric: 90 + 2 × age). |
| `temp_c` | float64 | °C, 34–41.5 | sampled | Normal, mean from 38.8 °C (ESI 1) down to 36.6 °C (ESI 5). |
| `has_high_risk_vitals` | int8 | 0 / 1 | derived | 1 if any ESI danger-zone vital: HR > 100 adult/geriatric (pediatric: > 140 under 5, > 110 age 5+); RR > 20 (pediatric: > 35 / > 24); SpO₂ < 92%. |
| `esi_v4_level` | int8 | ESI 1–5 | sampled | Hidden "true acuity", p = 0.023 / 0.271 / 0.417 / 0.270 / 0.019. Drives the vitals and outcomes. **Not a model feature.** |
| `esi_v5_level` | int8 | ESI 1–5 | derived | `esi_v4_level`, upgraded to 2 when ESI 3–5 has high-risk vitals (the ESI v5 danger-zone rule). |
| `resources_used` | int8 | count, 0–5 | sampled | Labs, imaging, IVs and consults, drawn from ranges typical for each `esi_v4_level`. |
| `current_wait_time_mins` | float64 | minutes | sampled | Exponential (mean 10 min for ESI 1–2, 45 min otherwise), × 3 during peak shift. |
| `is_peak_shift` | int8 | 0 / 1 | sampled | Peak-hours arrival (p = 0.25). |
| `override_occurred` | int8 | 0 / 1 | derived | 1 when a clinician would plausibly up-triage: frailty ≥ 6 with ESI ≥ 3, or no history + high-risk vitals + ESI ≥ 3. |
| `admitted_to_icu` | int8 | 0 / 1 | label | Bernoulli: 0.65 (ESI 1), 0.12 (ESI 2), 0.03 (+0.02 if frailty ≥ 5) for ESI 3, 0.005 for ESI 4–5. |
| `mortality_30d` | int8 | 0 / 1 | label | Bernoulli: 0.225 (ESI 1), 0.031 (+0.03 if frailty ≥ 5) for ESI 2, 0.016 (+0.02 if frailty ≥ 5) for ESI 3, 0.001 for ESI 4–5. |
| `critical_outcome` | int8 | 0 / 1 | label | `admitted_to_icu OR mortality_30d`. **The target all three models predict.** |
| `model_split` | string | `train` / `test` | meta | Whether the row trained its cohort's model or was held out for the reported metrics. |
| `data_origin` | string | `synthetic` | meta | Provenance flag. Every row is synthetic. |

### Features used by each model

| Model | Features |
|---|---|
| Geriatric | `heart_rate`, `resp_rate`, `spo2`, `sbp`, `temp_c`, `cfs_frailty_score`, `has_prior_history`, `comorbidity_count` |
| Adult | `heart_rate`, `resp_rate`, `spo2`, `sbp`, `temp_c`, `has_prior_history`, `comorbidity_count` |
| Pediatric | `age`, `heart_rate`, `resp_rate`, `spo2`, `sbp`, `temp_c`, `has_prior_history`, `comorbidity_count` |

The models never see `esi_v4_level`, `esi_v5_level` or any outcome column. They must infer risk from vitals and history, as a triage nurse would at the door.

## 5. How the data is generated

The generator works like a small causal model with a hidden acuity level:

```
age_cohort, age ──┐
                  ├──► vitals (HR, RR, SpO₂, SBP, temp) ──► has_high_risk_vitals ──► esi_v5_level
esi_v4_level ─────┤
(hidden acuity)   ├──► resources_used, wait time
                  └──► ICU admission, 30-day mortality ◄── cfs_frailty_score
                                    │
                                    └──► critical_outcome (target)
```

1. **Who arrives.** Draw cohort, age, gender, frailty and whether history is available.
2. **How sick they really are.** Draw a baseline ESI level (`esi_v4_level`).
3. **What the nurse measures.** Generate vitals around an age-appropriate baseline, pushed further out of range the more acute the patient. ESI 1 patients get either very high or very low heart and respiratory rates. ESI 3–4 patients have an 18% / 10% chance of one borderline-abnormal vital.
4. **What the rules say.** Apply the ESI v5 danger-zone thresholds to get `has_high_risk_vitals` and the upgraded `esi_v5_level`.
5. **What happens to them.** Draw ICU admission and 30-day death from acuity (and frailty for ESI 2–3), then combine them into `critical_outcome`.

The thresholds in step 4 are a subset of those the live rule engine uses. The full set in [`erflow/data/esi_reference_tables.json`](erflow/data/esi_reference_tables.json) also has infant ranges, blood-pressure limits and a lower heart-rate limit for frail older patients.

## 6. Known limitations

These are properties of the generator, not the real world. Read the model metrics with them in mind.

- **The metrics show feasibility, not clinical accuracy.** The ROC-AUC, sensitivity and specificity in the README measure how well a model recovers patterns *this generator put in*. They say nothing about performance on real patients.
- **Outcomes depend only on hidden acuity and frailty.** `comorbidity_count`, `has_prior_history`, `age` (within a cohort) and `gender` have **no effect** on outcomes in the generator. So any SHAP importance the models give them is noise, not a clinical signal.
- **Vitals are independent given acuity.** Real physiology links vitals together (e.g. fever raises heart rate; shock lowers blood pressure and raises heart rate). Here each vital is drawn separately.
- **No time structure.** There is no time of day, season or arrival sequence beyond the `is_peak_shift` flag. The data cannot be used to study patient flow over time.
- **No chief complaint, pain score or mental status.** The rule engine's ESI 1 and 2 checks (e.g. unresponsiveness) take these as UI inputs, but they are not in the dataset.
- **Pediatric frailty is a constant.** `cfs_frailty_score` is always 1 for children because the scale is not validated under 18. The pediatric model does not use it.
- **Distribution parameters are hand-set approximations.** Cohort mix, ESI mix, outcome rates and vital-sign shifts were chosen to look like a typical ED (e.g. an ESI 1 critical-outcome rate near 75%, and 8.6% critical overall). They were not fitted to a specific hospital's records.
- **Small positive class for children.** The pediatric test set has only 16 critical cases, so its metrics carry wide uncertainty.
