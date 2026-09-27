"""
Parquet Export for the ERflow Training Dataset
Writes the seeded synthetic ED cohort to erflow/data/erflow_ed_dataset.parquet with an
explicit schema, a train/test split column matching train_models.py, and provenance
metadata embedded in the file. Column meanings are documented in DATA.md.

Usage:  python -m erflow.scripts.export_parquet
"""

import os
import datetime as dt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from erflow.scripts.generate_data import generate_synthetic_ed_data

N_SAMPLES = 6230
RANDOM_SEED = 42

# Explicit types so the file is identical regardless of the pandas/pyarrow version used.
SCHEMA = pa.schema([
    ('patient_id', pa.string()),
    ('age', pa.int16()),
    ('age_cohort', pa.string()),
    ('gender', pa.string()),
    ('cfs_frailty_score', pa.int8()),
    ('has_prior_history', pa.int8()),
    ('comorbidity_count', pa.int8()),
    ('heart_rate', pa.float64()),
    ('resp_rate', pa.float64()),
    ('spo2', pa.float64()),
    ('sbp', pa.float64()),
    ('temp_c', pa.float64()),
    ('has_high_risk_vitals', pa.int8()),
    ('esi_v4_level', pa.int8()),
    ('esi_v5_level', pa.int8()),
    ('resources_used', pa.int8()),
    ('current_wait_time_mins', pa.float64()),
    ('is_peak_shift', pa.int8()),
    ('override_occurred', pa.int8()),
    ('admitted_to_icu', pa.int8()),
    ('mortality_30d', pa.int8()),
    ('critical_outcome', pa.int8()),
    ('model_split', pa.string()),
    ('data_origin', pa.string()),
])


def assign_model_split(df: pd.DataFrame) -> pd.Series:
    """
    Reproduce the per-cohort stratified 80/20 split used in train_models.py, so each row
    records whether it trained a model or was held out for the reported test metrics.
    """
    split = pd.Series('train', index=df.index)
    for cohort in ('geriatric', 'adult', 'pediatric'):
        cohort_df = df[df['age_cohort'] == cohort]
        _, test_idx = train_test_split(
            cohort_df.index, test_size=0.2,
            stratify=cohort_df['critical_outcome'], random_state=42
        )
        split[test_idx] = 'test'
    return split


def export_parquet(out_path: str = None) -> str:
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'erflow_ed_dataset.parquet')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = generate_synthetic_ed_data(n_samples=N_SAMPLES, random_seed=RANDOM_SEED)
    df['model_split'] = assign_model_split(df)
    df['data_origin'] = 'synthetic'

    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    table = table.replace_schema_metadata({
        'erflow.generator': 'erflow/scripts/generate_data.py::generate_synthetic_ed_data',
        'erflow.random_seed': str(RANDOM_SEED),
        'erflow.n_samples': str(N_SAMPLES),
        'erflow.exported_at_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'erflow.data_dictionary': 'DATA.md',
    })
    pq.write_table(table, out_path, compression='zstd')
    return out_path


if __name__ == '__main__':
    path = export_parquet()
    t = pq.read_table(path)
    print(f"Wrote {t.num_rows} rows x {t.num_columns} columns to {path}")
