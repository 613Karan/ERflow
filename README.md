# ERflow

**Emergency Department Decision Support System & Queue Scheduler**

**Team:** Argo (Karan Aditya, Jai A Mishra)  
**Institution:** Indian Institute of Technology (IIT) Guwahati  

---

## 1. Overview & Core Principles

**ERflow** is a two-stage decision support prototype for Emergency Department (ED) patient prioritization. It assesses each incoming patient, explains the result, and keeps the waiting room ordered by clinical priority, without replacing clinical judgment.

- **Asymmetric Risk Optimization**: Missing a critical case is far worse than over-prioritizing a minor one. Three age-specific XGBoost models are trained with an asymmetric loss that weights false negatives heavily ($\alpha = 23.0$ geriatric, $\alpha = 18.0$ adult, $\alpha = 28.0$ pediatric; $\beta = 1.0$).
- **Deterministic Safety Floors**: Model outputs are bounded by ESI v5 rule checks. Priority can only be escalated automatically, never downgraded without clinician action:
  $$\text{Final\_ESI} = \min(\text{ML\_ESI\_Recommendation}, \text{ABCDE\_ESI\_Floor})$$
- **Zero-History Resilience**: Built for realistic ED conditions where about half of patients arrive with no verified medical history.
- **Two-Stage Architecture**:
  - **Stage 1 (Patient Intake & Assessment)**: shows the final ESI level, risk probability, tree-level confidence score, triggered safety rules, SHAP feature contributions and a grounded clinical narrative, with a mandatory review gate for low-confidence cases.
  - **Stage 2 (Queue Scheduler)**: a continuously re-scored priority queue where scores update with waiting time:
    $$\text{Priority Score}(t) = W_{\text{floor}} \cdot (6 - \text{ESI}_{\text{final}}) + W_{\text{risk}} \cdot P_{\text{risk}} + W_{\text{time}} \cdot \ln\left(1 + \frac{t_{\text{wait}}}{\tau}\right)$$
    with $W_{\text{floor}} = 1000, W_{\text{risk}} = 100, W_{\text{time}} = 15, \tau = 30$.
- **Local-First Privacy**: models, SHAP, rules and the language model all run on the local machine.

---

## 2. Model Training

- **Dataset**: 6,230 synthetic ED patients generated with a fixed seed (`erflow/scripts/generate_data.py`, prototyped in `ERflow_Modelling.ipynb`). The target `critical_outcome` is ICU admission or 30-day mortality (536 positives, 8.6%).
- **Models**: one XGBoost model per age cohort, trained with a custom asymmetric logistic objective. Settings: 150 boosting rounds, `max_depth = 4`, `learning_rate = 0.01`, `tree_method = 'hist'`, stratified 80/20 train/test split, decision threshold `t = 0.504`.
- **Held-out test performance** (from `erflow/models/models_metadata.json`):

| Model | Rows (train / test) | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Geriatric (65+) | 1,829 (1,463 / 366) | 0.728 | 87.5% | 49.4% |
| Adult (18–64) | 3,431 (2,744 / 687) | 0.805 | 88.3% | 53.4% |
| Pediatric (<18) | 970 (776 / 194) | 0.695 | 75.0% | 52.2% |

Retrain with `python -m erflow.scripts.train_models`. Because the data is seeded, retraining reproduces these models.

---

## 3. Local Language Model (Gemma 4 via Ollama)

The Stage 1 clinical narrative is written by a local Gemma 4 model served by Ollama (`erflow/agent/llm_client.py`). The model only explains results already computed by the models and rules, and every narrative is checked by the grounding validator before display.

- Default model `gemma4:12b` at `http://localhost:11434` (`POST /api/chat`), overridable with `OLLAMA_MODEL` and `OLLAMA_HOST`.
- Requests are non-streaming, JSON-formatted, temperature 0.1, 15 s timeout.
- If Ollama is not running or a request fails, ERflow automatically uses a built-in deterministic narrative engine, so it works fully offline.

```bash
ollama pull gemma4:12b
ollama serve
```

---

## 4. Repository Structure

```
erflow/
├── models/
│   ├── geriatric_xgb.json       # Frailty-aware XGBoost model (65+, alpha=23.0)
│   ├── adult_xgb.json           # Acute derangement XGBoost model (18-64, alpha=18.0)
│   ├── pediatric_xgb.json       # Pediatric risk XGBoost model (<18, alpha=28.0)
│   └── models_metadata.json     # Features, thresholds and test metrics
├── data/
│   └── esi_reference_tables.json # ESI v5 vital thresholds & age reference ranges
├── rule_engine/
│   ├── decision_a.py            # Immediate life threat check (ESI 1)
│   ├── decision_b.py            # Altered mental status / high risk check (ESI 2)
│   ├── decision_c.py            # Resource estimation & fallback handler
│   ├── decision_d.py            # Danger zone vital check (ESI 2) + Geriatric Frailty Guard
│   └── engine.py                # ABCDE safety floor aggregation
├── scheduler/
│   ├── scoring.py               # Priority Score continuous formula
│   └── max_heap.py              # Priority queue manager & override audit log
├── explain/
│   ├── shap_explainer.py        # SHAP TreeExplainer extraction
│   ├── uncertainty.py           # Tree-level ensemble variance score
│   └── grounding.py             # Structured JSON claim validator
├── agent/
│   ├── tools.py                 # Model, SHAP, rule and reference-range tools + schemas
│   ├── orchestrator.py          # Assessment pipeline & narrative generation
│   └── llm_client.py            # Ollama (Gemma 4) client & deterministic fallback
├── api/
│   └── main.py                  # FastAPI endpoints (/analyze-patient, /queue, /override, ...)
├── ui/
│   └── app.py                   # Streamlit dashboard
├── scripts/
│   ├── generate_data.py         # Synthetic ED dataset generator (6,230 cases)
│   ├── train_models.py          # Asymmetric loss model trainer & exporter
│   ├── generate_documentation_pdf.py  # Builds ERflow_Technical_Documentation.pdf
│   └── generate_mismatch_pdf.py       # Builds ERflow_ESI_Disagreement_and_Conflict_Resolution.pdf
└── tests/
    ├── test_rule_engine.py      # ESI handbook practice test cases
    ├── test_scheduler.py        # Priority scoring & queue tests
    ├── test_grounding.py        # Hallucination rejection unit tests
    └── test_api.py              # API endpoint integration tests
```

---

## 5. Quick Start

### Run the Streamlit Dashboard
```bash
python main.py --ui
```

### Run the FastAPI Backend
```bash
python main.py --api
# API documentation available at http://localhost:8000/docs
```

### Run the Terminal Batch Simulation
```bash
python main.py --simulate
```

### Run the Test Suite
```bash
python -m pytest erflow/tests -v
```

---

## 6. Clinical Governance
ERflow provides risk estimates, safety floors and grounded explanations to support clinicians. It keeps predictive risk insights separate from diagnosis, and clinicians keep final authority to override any acuity assignment. Every override is recorded in the Nurse Override Audit Log.
