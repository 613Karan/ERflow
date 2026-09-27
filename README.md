# ERflow

> ### Live demo: **[erflow-qcbv7gnpmrgbbvxcqd2gqw.streamlit.app](https://erflow-qcbv7gnpmrgbbvxcqd2gqw.streamlit.app/)**
> No install needed. If the page shows "Zzzz… this app has gone to sleep", click **Yes, get this app back up** and give it about a minute.
>
> **Dataset:** [`erflow/data/erflow_ed_dataset.parquet`](erflow/data/erflow_ed_dataset.parquet) · schema and provenance in **[DATA.md](DATA.md)**

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

- **Dataset**: 6,230 synthetic ED patients generated with a fixed seed (`erflow/scripts/generate_data.py`, prototyped in `ERflow_Modelling.ipynb`), exported to [`erflow/data/erflow_ed_dataset.parquet`](erflow/data/erflow_ed_dataset.parquet) and documented in [DATA.md](DATA.md). The target `critical_outcome` is ICU admission or 30-day mortality (536 positives, 8.6%).
- **Models**: one XGBoost model per age cohort, trained with a custom asymmetric logistic objective. Settings: 150 boosting rounds, `max_depth = 4`, `learning_rate = 0.01`, `tree_method = 'hist'`, stratified 80/20 train/test split, decision threshold `t = 0.504`.
- **Held-out test performance** (from `erflow/models/models_metadata.json`):

| Model | Rows (train / test) | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Geriatric (65+) | 1,829 (1,463 / 366) | 0.728 | 87.5% | 49.4% |
| Adult (18–64) | 3,431 (2,744 / 687) | 0.805 | 88.3% | 53.4% |
| Pediatric (<18) | 970 (776 / 194) | 0.695 | 75.0% | 52.2% |

Retrain with `python -m erflow.scripts.train_models`. Because the data is seeded, retraining reproduces these models.

---

## 3. What the Metrics Mean

**These numbers measure feasibility, not clinical accuracy.** The test sets are held-out rows of the *same synthetic generator* the models were trained on (see [DATA.md](DATA.md)). They show that the pipeline can recover a risk signal from triage vitals. They do not predict how ERflow would perform on real patients, which would need validation on real ED records.

**Confusion matrices on the held-out test sets** (95% Wilson confidence intervals):

| Model | Critical caught (TP) | Critical missed (FN) | Over-flagged (FP) | Correctly cleared (TN) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---:|---:|---:|---:|---|---|
| Geriatric | 28 | 4 | 169 | 165 | 87.5% (72–95%) | 49.4% (44–55%) |
| Adult | 53 | 7 | 292 | 335 | 88.3% (78–94%) | 53.4% (50–57%) |
| Pediatric | 12 | 4 | 85 | 93 | 75.0% (51–90%) | 52.2% (45–59%) |

**Why we accept low specificity.** In triage the two errors do not cost the same:
- A **false negative** means a critically ill patient waits as if they were routine. That can mean deterioration in the waiting room or death.
- A **false positive** means a stable patient is looked at sooner than necessary. That costs a few minutes of a nurse's attention.

So the models are trained with an asymmetric loss that penalises a missed critical case 18–28× more than a false alarm (α in §1). The price is that about half of all patients get flagged. The flag is therefore a screening signal, not a verdict:
- The queue orders patients by the continuous priority score (ESI floor + risk probability + waiting time), not by the yes/no flag.
- The ESI v5 rule engine sets a floor that the model can raise but never lower, so a missed case with danger-zone vitals is still escalated by the rules.
- Cases where the trees disagree (confidence score < 20%) are sent to mandatory clinician review.

**The pediatric numbers are the least certain.** Its test set has only 16 critical cases, so sensitivity could lie anywhere from about 51% to 90%.

---

## 4. Scope: Understand → Predict → Recommend → Actuate

| Level | ERflow | How |
|---|---|---|
| **Understand** | Yes | SHAP contributions, triggered safety rules and a grounded narrative show *why* a patient is high-risk. |
| **Predict** | Yes | Age-specific XGBoost models estimate the probability of ICU admission or 30-day death. |
| **Recommend** | Yes | A final ESI level (the model result bounded by rule floors) and a continuously re-ranked waiting-room queue tell staff who to see next. |
| **Actuate** | No (by design) | ERflow never acts on its own. A clinician confirms or overrides every acuity level, and every override is logged. |

---

## 5. Limitations: What ERflow Cannot Tell You

- **How it performs on real patients.** Every model was trained and tested on synthetic data (§3).
- **Which history factors matter.** In the generator, comorbidities, prior history and gender have no effect on outcomes. Any SHAP weight the models give them is noise, not a learned clinical signal ([DATA.md §6](DATA.md#6-known-limitations)).
- **Anything beyond vitals and frailty.** There is no chief complaint, pain score, medication list, lab result or free-text note. The ESI 1 and 2 rule checks (e.g. unresponsiveness) rely on the nurse's input in the UI.
- **Patient flow over time.** There are no real arrival times, seasons or staffing levels, so ERflow cannot forecast crowding or wait times.
- **Diagnosis.** ERflow estimates *risk of a critical outcome*. It does not say what is wrong with the patient.
- **Whether the queue weights are optimal.** The priority-score weights (W_floor = 1000, W_risk = 100, W_time = 15, τ = 30) are design choices that make a lower ESI always outrank a higher one while letting waiting time break ties. They were not tuned against real outcomes.
- **Fairness across groups.** The synthetic data has no ethnicity, language or socioeconomic fields, so bias against real sub-populations cannot be measured here.

---

## 6. Privacy, Safety & Clinical Governance

**Privacy**
- **No real patient data.** The dataset is fully synthetic. No person was observed, recorded or identified at any point.
- **Local-first.** The models, SHAP, rule engine and Gemma all run on the local machine. In a real deployment no patient information would leave the hospital network. The public Streamlit demo only processes values typed into the form, and it stores nothing.

**Safety**
- **The model can only escalate.** `Final_ESI = min(ML_ESI, Rule_Floor)`. A deterministic ESI v5 rule check sets a floor the model cannot lower.
- **Uncertainty is surfaced.** A tree-level confidence score is shown with every result. Below 20%, a mandatory review gate stops the result being accepted without a clinician.
- **The language model cannot decide anything.** Gemma only explains a result that is already computed. A grounding validator rejects any narrative whose ESI, risk factors or safety rules don't match the computed values, and ERflow falls back to a deterministic narrative engine.

**Clinical governance**
ERflow provides risk estimates, safety floors and grounded explanations to support clinicians. It keeps risk estimates separate from diagnosis, and clinicians keep final authority to override any acuity assignment. Every override is recorded, with clinician ID and reason, in the Nurse Override Audit Log.

---

## 7. Local Language Model (Gemma 4 via Ollama)

The Stage 1 clinical narrative is written by a local Gemma 4 model served by Ollama (`erflow/agent/llm_client.py`). The model only explains results already computed by the models and rules, and every narrative is checked by the grounding validator before display.

- Default model `gemma4:12b` at `http://localhost:11434` (`POST /api/chat`), overridable with `OLLAMA_MODEL` and `OLLAMA_HOST`.
- Requests are non-streaming, JSON-formatted, temperature 0.1, 90 s timeout (`OLLAMA_TIMEOUT`).
- If Ollama is not running or a request fails, ERflow automatically uses a built-in deterministic narrative engine, so it works fully offline.

```bash
ollama pull gemma4:12b
ollama serve
```

---

## 8. Repository Structure

```
erflow/
├── models/
│   ├── geriatric_xgb.json       # Frailty-aware XGBoost model (65+, alpha=23.0)
│   ├── adult_xgb.json           # Acute derangement XGBoost model (18-64, alpha=18.0)
│   ├── pediatric_xgb.json       # Pediatric risk XGBoost model (<18, alpha=28.0)
│   └── models_metadata.json     # Features, thresholds and test metrics
├── data/
│   ├── erflow_ed_dataset.parquet # Training dataset (6,230 rows), see DATA.md
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
│   ├── export_parquet.py        # Writes the training dataset to Parquet
│   ├── generate_documentation_pdf.py  # Builds ERflow_Technical_Documentation.pdf
│   └── generate_mismatch_pdf.py       # Builds ERflow_ESI_Disagreement_and_Conflict_Resolution.pdf
└── tests/
    ├── test_rule_engine.py      # ESI handbook practice test cases
    ├── test_scheduler.py        # Priority scoring & queue tests
    ├── test_grounding.py        # Hallucination rejection unit tests
    └── test_api.py              # API endpoint integration tests
```

---

## 9. Quick Start

### Install Dependencies
The `.venv` committed in earlier revisions was built on macOS and is not portable. Create your own:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt                          # to run the app
pip install -r requirements.txt -r requirements-dev.txt   # to also run tests / build PDFs
```

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

## 10. Deployment

The dashboard calls the assessment pipeline in-process and never contacts the FastAPI backend, so it
deploys as a **single service**. Two things ship alongside the code to make that work:
`requirements.txt` (pinned versions) and `runtime.txt` (Python 3.12, chosen for the widest
`xgboost` / `shap` wheel coverage). Streamlit Community Cloud does not read `runtime.txt`, so
the version is set in its dashboard instead (step 4).

### Streamlit Community Cloud (recommended)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → pick this repository, branch `main`.
3. Set **Main file path** to `erflow/ui/app.py`.
4. Open **Advanced settings** and set **Python version** to **3.12**. On newer versions some pinned
   packages have no ready-made wheels, so they compile from source and the build can stall for 30+ minutes.
   The version can't be changed after deployment; only a delete and redeploy changes it.
5. Deploy. The first build takes a few minutes.

The loaded app holds roughly 370 MB resident, comfortably inside the free tier.

### GitHub Pages

GitHub Pages serves static files only and **cannot run this app** — every interaction executes
XGBoost inference and SHAP server-side. What Pages *can* host is the project landing page in
`docs/index.html`, which describes the system and links to the live dashboard:

1. **Settings → Pages → Source:** Deploy from a branch.
2. **Branch:** `main`, **folder:** `/docs`. Save.
3. The page publishes at `https://613Karan.github.io/ERflow/`.

Edit the "Launch the live dashboard" link in `docs/index.html` to point at your Streamlit URL once
step 5 above completes.

### Other hosts

Hugging Face Spaces (Streamlit SDK), Render and Railway all work from the same
`requirements.txt`; start the app with:

```bash
streamlit run erflow/ui/app.py --server.port $PORT --server.address 0.0.0.0
```

### The local language model does not deploy

`gemma4:12b` is 8.3 GB resident and wants a GPU, so no free tier can host it. A deployed instance
uses the built-in deterministic narrative engine: instant, fully grounded, and labelled as such in
the Stage 1 UI. Gemma remains a local-only enhancement. To force the deterministic path locally —
useful for a fast demo — point `OLLAMA_HOST` at a closed port:

```bash
OLLAMA_HOST=http://127.0.0.1:1 python main.py --ui
```

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `gemma4:12b` | Model tag to request |
| `OLLAMA_TIMEOUT` | `90` | Seconds before falling back |

---

## 11. Challenges & How We Solved Them

| Problem | What happened | What we did |
|---|---|---|
| **Gemma timing out** | `gemma4:12b` spent its time in a hidden "thinking" phase and missed the request timeout. Passing `think: false` inside `options` was silently ignored by Ollama. | Sent `think: false` as a top-level request field. If an Ollama build rejects it (HTTP 400), the client retries without it automatically. |
| **Empty or malformed LLM answers** | Sometimes the answer landed in the thinking channel, leaving the content empty, or came back as invalid JSON. | Every LLM response must parse as JSON and contain all four required fields, then pass the grounding validator. Anything else falls back to the deterministic narrative engine, so the UI never shows an unchecked explanation. |
| **The LLM can't be hosted for free** | Gemma needs about 8.3 GB of memory and ideally a GPU. No free host provides that. | The deployed demo uses the deterministic narrative engine and labels it as such. Gemma stays a local enhancement (§7). |

---

## 12. Development Timeline

Everything in this repository was built during the 48-hour hackathon window: the data generator, dataset, models, rule engine, scheduler, explainability layer, dashboard, API, tests and documentation. No code, data or models were brought in from before the event. The commit history on `main` shows the work in the order it happened.
