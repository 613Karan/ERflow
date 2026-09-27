"""
Script to generate the ERflow Technical System Documentation PDF.
Covers the synthetic dataset, demographic XGBoost model training, the inference and explainability
pipeline, the ESI v5 safety floor engine, the local Gemma 4 integration via Ollama, the queue
scheduler, application interfaces, and the verification suite.

Usage:
    python -m erflow.scripts.generate_documentation_pdf
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted, KeepTogether
)

from erflow.scripts.generate_mismatch_pdf import NumberedCanvas


class DocumentationCanvas(NumberedCanvas):
    """
    Running header/footer for the technical documentation (same layout as the disagreement guide).
    """
    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "ERflow — Technical System Documentation & Architecture Guide")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Engineering Reference")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            self.line(54, 45, letter[0] - 54, 45)
            self.drawString(54, 32, "Confidential — Team Argo (Karan Aditya, Jai A Mishra) | IIT Guwahati")
            self.drawRightString(letter[0] - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def _load_metadata() -> dict:
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'models_metadata.json')
    with open(meta_path, 'r') as f:
        return json.load(f)


def generate_technical_documentation_pdf(output_pdf_path: str):
    meta = _load_metadata()

    doc = SimpleDocTemplate(
        output_pdf_path, pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54,
        title="ERflow — Technical System Documentation", author="Team Argo"
    )
    styles = getSampleStyleSheet()

    primary_blue = colors.HexColor("#1E3A8A")
    teal = colors.HexColor("#0F766E")
    dark_slate = colors.HexColor("#1F2937")
    light_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold',
                                 fontSize=24, leading=28, textColor=primary_blue, spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSubTitle', parent=styles['Normal'], fontName='Helvetica',
                                    fontSize=11.5, leading=15, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h1 = ParagraphStyle('H1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13.5, leading=17,
                        textColor=primary_blue, spaceBefore=12, spaceAfter=5, keepWithNext=True)
    h2 = ParagraphStyle('H2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, leading=14,
                        textColor=teal, spaceBefore=7, spaceAfter=3, keepWithNext=True)
    body = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13,
                          textColor=dark_slate, spaceAfter=5)
    bullet = ParagraphStyle('Bullet', parent=body, leftIndent=12, firstLineIndent=-8, spaceAfter=3)
    callout = ParagraphStyle('Callout', parent=styles['Normal'], fontName='Helvetica', fontSize=8.8, leading=12.5,
                             textColor=colors.HexColor("#1E293B"))
    th = ParagraphStyle('TH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10.5,
                        textColor=colors.white)
    td = ParagraphStyle('TD', parent=styles['Normal'], fontName='Helvetica', fontSize=7.8, leading=10.5,
                        textColor=dark_slate)
    code = ParagraphStyle('Code', parent=styles['Normal'], fontName='Courier', fontSize=7.6, leading=10,
                          textColor=colors.HexColor("#0F172A"))

    def table(rows, widths, header_bg=primary_blue):
        data = [[Paragraph(c, th) for c in rows[0]]] + [[Paragraph(str(c), td) for c in r] for r in rows[1:]]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 4.5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ]))
        return t

    def box(flowable, bg=light_bg, border=colors.HexColor("#CBD5E1")):
        t = Table([[flowable]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg),
            ('BOX', (0, 0), (-1, -1), 0.8, border),
            ('PADDING', (0, 0), (-1, -1), 7),
        ]))
        return t

    def code_block(text):
        return box(Preformatted(text, code))

    def bullets(items):
        return [Paragraph("• " + i, bullet) for i in items]

    story = []

    # ------------------------------------------------------------------
    # TITLE
    # ------------------------------------------------------------------
    story.append(Paragraph("ERflow", title_style))
    story.append(Paragraph("Technical System Documentation &amp; Architecture Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_blue, spaceBefore=0, spaceAfter=8))
    meta_rows = [
        [Paragraph("<b>Project:</b> ERflow — Emergency Department Decision Support Prototype", body),
         Paragraph("<b>Institution:</b> Indian Institute of Technology (IIT) Guwahati", body)],
        [Paragraph("<b>Team:</b> Argo (Karan Aditya, Jai A Mishra)", body),
         Paragraph("<b>Deployment:</b> Local-first — all inference runs on-device; no patient data leaves the machine", body)],
    ]
    mt = Table(meta_rows, colWidths=[252, 252])
    mt.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), light_bg), ('PADDING', (0, 0), (-1, -1), 5),
                            ('BOX', (0, 0), (-1, -1), 0.5, border_color), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    story.append(mt)
    story.append(Spacer(1, 8))
    story.append(box(Paragraph(
        "<b>Executive Summary:</b> ERflow is a two-stage decision support system for Emergency Department (ED) patient "
        "prioritization. Stage 1 assesses each incoming patient with one of three age-specific XGBoost risk models "
        "trained with an asymmetric (false-negative-averse) loss, bounds the result with a deterministic ESI Version 5 "
        "safety floor, explains it with SHAP attributions and a tree-level confidence score, and writes a grounded "
        "clinical narrative using a local Gemma 4 model served by Ollama. Stage 2 places the patient in a continuously "
        "re-scored priority queue with full clinician override and audit logging.", callout),
        bg=colors.HexColor("#EFF6FF"), border=colors.HexColor("#93C5FD")))

    # ------------------------------------------------------------------
    # 1. PRINCIPLES
    # ------------------------------------------------------------------
    story.append(Paragraph("1. Core Principles", h1))
    story += bullets([
        "<b>Asymmetric Risk Optimization:</b> missing a critical patient (false negative) is far costlier than "
        "over-prioritizing a stable one, so training penalizes false negatives 18–28× more than false positives.",
        "<b>Deterministic Safety Floors:</b> Final_ESI = min(ML_ESI_Recommendation, ABCDE_ESI_Floor). Rules can only "
        "escalate a patient; the model can never downgrade one below the floor.",
        "<b>Zero-History Resilience:</b> about half of ED arrivals have no verified records, so the dataset and models "
        "are built with a 50% zero-history share and rely on presenting vitals and frailty.",
        "<b>Human-in-the-Loop:</b> low-confidence predictions trigger a mandatory review gate, and nurses can override "
        "any acuity level with a recorded rationale.",
        "<b>Local-First Privacy:</b> models, SHAP, rules and the language model all run on the local machine.",
    ])

    # ------------------------------------------------------------------
    # 2. DATASET
    # ------------------------------------------------------------------
    story.append(Paragraph("2. Synthetic Training Dataset (scripts/generate_data.py)", h1))
    story.append(Paragraph(
        "Models are trained on a reproducible synthetic ED cohort produced by <font face='Courier'>generate_synthetic_ed_data()</font> "
        "(the same generator is prototyped in <font face='Courier'>ERflow_Modelling.ipynb</font>). Default size is "
        "<b>6,230 patients</b> with random seed <b>42</b>, so every run yields identical data.", body))
    story.append(table([
        ["Component", "Generation Logic"],
        ["Age cohorts", "Pediatric 15% (ages 1–17), Adult 55% (18–64), Geriatric 30% (65–94). Realised: 970 / 3,431 / 1,829."],
        ["Gender", "Female 47%, Male 53%."],
        ["Clinical Frailty Scale", "Pediatric fixed at 1. Adult CFS 1–4 (p = 0.50/0.30/0.15/0.05). Geriatric CFS 1–9 "
                                   "(p = 0.05/0.15/0.25/0.20/0.15/0.10/0.05/0.03/0.02)."],
        ["Prior history", "50% verified records, 50% zero-history. Comorbidity count ~ Poisson(λ = 2.5 geriatric, 0.8 otherwise) "
                          "when history exists, else 0."],
        ["Baseline acuity (ESI v4)", "ESI 1–5 drawn with p = 0.023 / 0.271 / 0.417 / 0.270 / 0.019."],
        ["Vital signs", "Generated from the baseline acuity around cohort baselines (adult HR 75, RR 16, SBP 120; geriatric "
                        "HR 72, RR 18, SBP 135; pediatric HR 110/90, RR 28/20 below/above age 5, SBP 90 + 2×age). ESI 1 "
                        "draws unstable extremes; ESI 3 and 4 carry an 18% / 10% chance of one abnormal vital. Values are "
                        "clipped to HR 30–220, RR 4–60, SpO2 60–100, SBP 50–240, Temp 34–41.5 °C."],
        ["ESI v5 level", "ESI 3–5 patients with any age-specific danger-zone vital (HR, RR or SpO2 &lt; 92%) are escalated to ESI 2."],
        ["Outcome label", "<b>critical_outcome</b> = ICU admission OR 30-day mortality. ICU probability 0.65 / 0.12 / 0.03 "
                          "(+0.02 if CFS ≥ 5) / 0.005 for ESI 1 / 2 / 3 / 4–5; mortality 0.225 / 0.031 (+0.03 if CFS ≥ 5) / "
                          "0.016 (+0.02 if CFS ≥ 5) / 0.001. Result: 536 positives (8.6%)."],
        ["Queue features", "Wait time ~ Exponential(scale 10 min for ESI 1–2, 45 min otherwise), tripled for the 25% of "
                           "patients on peak-hour shifts. Also records resources used and override occurrence."],
    ], [110, 394]))

    # ------------------------------------------------------------------
    # 3. MODEL TRAINING
    # ------------------------------------------------------------------
    story.append(Paragraph("3. Demographic XGBoost Model Training (scripts/train_models.py)", h1))
    story.append(Paragraph(
        "A single adult-calibrated model is unsafe across age groups because physiological baselines differ, so ERflow "
        "trains three independent gradient-boosted models, one per cohort. Each cohort is split 80/20 into train and "
        "test sets, stratified on the outcome label (random_state = 42).", body))
    story.append(Paragraph("3.1 Training Configuration", h2))
    story.append(table([
        ["Setting", "Value"],
        ["Library / API", "XGBoost native API: xgb.train() on xgb.DMatrix inputs"],
        ["Boosting rounds", "150"],
        ["Tree parameters", "max_depth = 4, learning_rate = 0.01, tree_method = 'hist', min_child_weight = 1"],
        ["Objective", "Custom asymmetric logistic loss (below); built-in eval metric disabled (disable_default_eval_metric = 1)"],
        ["Decision threshold", "t = 0.504 on P_risk = sigmoid(raw margin), shared by all three models"],
        ["Export", "Booster saved as JSON: models/geriatric_xgb.json, adult_xgb.json, pediatric_xgb.json; "
                   "evaluation summary written to models/models_metadata.json"],
    ], [110, 394]))
    story.append(Paragraph("3.2 Asymmetric Logistic Objective", h2))
    story.append(Paragraph(
        "For label y ∈ {0, 1} and p = 1 / (1 + e<super>−margin</super>), the loss is "
        "L = −[α·y·log(p) + β·(1 − y)·log(1 − p)] with β = 1.0. XGBoost receives its first and second derivatives:", body))
    story.append(code_block(
        "grad = p * (alpha * y + beta * (1 - y)) - alpha * y\n"
        "hess = p * (1 - p) * (alpha * y + beta * (1 - y))"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.3 Per-Cohort Models and Held-Out Test Performance", h2))
    split = {'geriatric': (1829, 1463, 366, 158), 'adult': (3431, 2744, 687, 298), 'pediatric': (970, 776, 194, 80)}
    labels = {'geriatric': "Geriatric (65+)", 'adult': "Adult (18–64)", 'pediatric': "Pediatric (&lt;18)"}
    feat_text = {
        'geriatric': "HR, RR, SpO2, SBP, Temp, CFS, prior history, comorbidity count",
        'adult': "HR, RR, SpO2, SBP, Temp, prior history, comorbidity count",
        'pediatric': "Age, HR, RR, SpO2, SBP, Temp, prior history, comorbidity count",
    }
    rows = [["Model", "α", "Rows (train / test)", "Critical", "Features", "AUC", "Sens.", "Spec.", "TP / FN / FP / TN"]]
    for c in ['geriatric', 'adult', 'pediatric']:
        m = meta[c]; n, tr, te, pos = split[c]
        rows.append([f"<b>{labels[c]}</b>", f"{m['alpha']:.1f}", f"{n:,} ({tr:,} / {te})", str(pos), feat_text[c],
                     f"{m['roc_auc']:.3f}", f"{m['sensitivity']*100:.1f}%", f"{m['specificity']*100:.1f}%",
                     f"{m['tp']} / {m['fn']} / {m['fp']} / {m['tn']}"])
    story.append(table(rows, [58, 34, 66, 40, 112, 34, 40, 40, 80]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "α = false-negative weight in the loss; Critical = patients with the critical outcome; AUC = ROC-AUC; "
        "Sens. / Spec. = sensitivity / specificity at t = 0.504 on the held-out test set.", td))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Test-set figures are recomputed from the shipped model files and match models_metadata.json. The operating point "
        "deliberately trades specificity for sensitivity: across the three test sets 93 of 108 critical patients are "
        "flagged. Retrain with <font face='Courier'>python -m erflow.scripts.train_models</font>; because the dataset is "
        "seeded, retraining reproduces these models.", body))

    # ------------------------------------------------------------------
    # 4. INFERENCE & EXPLAINABILITY
    # ------------------------------------------------------------------
    story.append(Paragraph("4. Inference, Uncertainty &amp; Explainability", h1))
    story += bullets([
        "<b>Cohort routing:</b> the patient's age cohort field selects the model; if absent, age &lt; 18 → pediatric, "
        "age ≥ 65 → geriatric, otherwise adult. Missing inputs default to CFS 2, prior history 0, comorbidities 0.",
        "<b>ML recommendation (agent/tools.py):</b> P_risk ≥ 0.504 → ESI 2 (escalate); otherwise ESI 3 (standard).",
        "<b>Tree-level confidence (explain/uncertainty.py):</b> P_risk is re-predicted using only the first 15, 30, … 150 "
        "trees. The standard deviation of these 10 stage predictions is normalised by 0.25 and mapped to a 0–100% "
        "confidence score. Below 20% the patient is flagged for mandatory human review.",
        "<b>SHAP attribution (explain/shap_explainer.py):</b> shap.TreeExplainer computes exact Shapley values on the "
        "cohort's model; the top 5 features by |SHAP| are returned with direction and age-banded normal ranges from "
        "data/esi_reference_tables.json (infant, toddler/young child, school age/adolescent, adult, geriatric).",
        "<b>Grounding validator (explain/grounding.py):</b> every narrative is checked before display — claimed ESI must "
        "not be less urgent than the safety floor, each cited risk factor must match its SHAP sign, cited values must "
        "match intake data within ±0.5, cited features must exist in the model's feature space, and cited rules must "
        "have actually fired. Any violation marks the narrative ungrounded and forces human review.",
    ])

    # ------------------------------------------------------------------
    # 5. SAFETY FLOOR
    # ------------------------------------------------------------------
    story.append(Paragraph("5. Deterministic ESI v5 Safety Floor Engine (rule_engine/)", h1))
    story.append(table([
        ["Decision Point", "Clinical Logic", "Acuity Floor"],
        ["<b>A</b> (decision_a.py)", "Immediate life-saving intervention, cardiac arrest / apneic / pulseless, intubation or "
                                     "agonal respiration / airway compromise, AVPU = U or GCS ≤ 8, HR &lt; 30 or &gt; 220, "
                                     "SBP &lt; 50.", "ESI 1 (hard locked)"],
        ["<b>B</b> (decision_b.py)", "High-risk situation, altered mental status (AVPU V/P or GCS 9–13), severe pain/distress "
                                     "(or pain ≥ 8/10 with systemic distress), ischemic chest pain, stroke symptoms (FAST+), "
                                     "evolving anaphylaxis, acute suicidal ideation.", "ESI 2"],
        ["<b>D</b> (decision_d.py)", "Danger-zone vitals — infant &lt;1 y: HR &gt; 160, RR &gt; 45; age 1–4: HR &gt; 140, RR &gt; 35; "
                                     "age 5–17: HR &gt; 110, RR &gt; 24; all pediatric: SpO2 &lt; 92%, fever ≥ 38 °C under 3 months. "
                                     "Adult/geriatric: HR &gt; 100, RR &gt; 20, SpO2 &lt; 92%, SBP &lt; 90. Geriatric frailty guard: "
                                     "age ≥ 65, CFS ≥ 5 and HR &gt; 90.", "ESI 2"],
        ["<b>C</b> (decision_c.py)", "Expected resources: 2+ → ESI 3, 1 → ESI 4, 0 → ESI 5. If resources were not collected, "
                                     "returns insufficient_data and defaults to ESI 3.", "ESI 3 / 4 / 5"],
    ], [82, 332, 90], header_bg=teal))

    # ------------------------------------------------------------------
    # 6. LOCAL LLM
    # ------------------------------------------------------------------
    story.append(Paragraph("6. Local Language Model: Gemma 4 via Ollama (agent/llm_client.py)", h1))
    story.append(Paragraph(
        "The clinical narrative shown in Stage 1 is written by a local Gemma 4 model served by Ollama. The model never "
        "decides acuity — it only explains results already computed by the models and rules, and its output is "
        "validated by the grounding validator before display.", body))
    story.append(table([
        ["Setting", "Value"],
        ["Default model", "gemma4:12b (override with the OLLAMA_MODEL environment variable)"],
        ["Server", "http://localhost:11434 (override with OLLAMA_HOST); endpoint POST /api/chat"],
        ["Availability check", "One TCP connect to 127.0.0.1:11434 with a 50 ms timeout when the client starts"],
        ["Request options", "stream = false, format = \"json\", temperature = 0.1, think = false (passed inside options)"],
        ["Timeout", "15 seconds per request"],
        ["Input", "User message = JSON of the patient record, ML result, rule result, SHAP result and final ESI"],
        ["Output handling", "message.content is parsed as JSON and returned as the structured narrative (provider = ollama)"],
    ], [100, 404]))
    story.append(Spacer(1, 5))
    story.append(Paragraph("System prompt sent with every request:", body))
    story.append(code_block(
        "You are ERflow, a clinical decision support assistant for emergency department patient\n"
        "prioritization. Synthesize a factual, grounded clinical acuity assessment based strictly on\n"
        "the provided ML predictions, SHAP feature rankings, and ESI v5 deterministic rules. Do not\n"
        "hallucinate. Return a structured JSON summary."))
    story.append(Spacer(1, 5))
    story.append(Paragraph("6.1 Deterministic Fallback", h2))
    story.append(Paragraph(
        "If Ollama is not running, the request fails, times out or returns invalid JSON, ERflow switches automatically "
        "to a built-in deterministic narrative engine (provider = deterministic_clinical_engine, model = "
        "esi_v5_grounded_synthesizer), so the system works fully offline with no setup. The fallback emits the fields "
        "the dashboard and validator consume: <font face='Courier'>clinical_rationale</font>, "
        "<font face='Courier'>alignment_status</font>, <font face='Courier'>primary_risk_factors</font> (feature, value, "
        "direction, SHAP impact), <font face='Courier'>triggered_safety_rules</font>, risk and confidence percentages, "
        "and the review flag.", body))
    story.append(Paragraph("6.2 Tool Schemas", h2))
    story.append(Paragraph(
        "agent/tools.py also exposes Ollama-compatible function schemas for four tools — predict_esi_xgboost, "
        "get_shap_explanation, run_esi_rule_engine and lookup_reference_range. In the current pipeline the orchestrator "
        "calls these tools directly in Python and passes their results to the model in a single request.", body))
    story.append(Paragraph("6.3 Setup", h2))
    story.append(code_block(
        "ollama pull gemma4:12b      # download the model once\n"
        "ollama serve                # start the local server on port 11434\n"
        "python main.py --ui         # ERflow detects the server at startup"))

    # ------------------------------------------------------------------
    # 7. ORCHESTRATION
    # ------------------------------------------------------------------
    story.append(Paragraph("7. Assessment Pipeline (agent/orchestrator.py)", h1))
    story.append(Paragraph("AssessmentOrchestrator.analyze_patient() runs these steps for every patient:", body))
    story += bullets([
        "Run the cohort's XGBoost model → P_risk, ML ESI, confidence score, review flag.",
        "Run the ESI v5 rule engine → acuity floor and triggered rules.",
        "Apply the safety floor: Final_ESI = min(ML ESI, floor); record the agreement state "
        "(AGREEMENT, RULE_SAFETY_ESCALATION or ML_PREDICTIVE_ESCALATION).",
        "Compute SHAP attributions (top 5) and the Stage 2 priority score.",
        "Generate the narrative (Gemma 4 via Ollama, or the deterministic fallback) and validate it; an ungrounded "
        "narrative forces human review.",
    ])

    # ------------------------------------------------------------------
    # 8. QUEUE SCHEDULER
    # ------------------------------------------------------------------
    story.append(Paragraph("8. Stage 2: Queue Scheduler (scheduler/)", h1))
    story.append(code_block(
        "Priority Score(t) = W_floor x (6 - ESI_final) + W_risk x P_risk + W_time x ln(1 + t_wait / tau)\n"
        "W_floor = 1000   W_risk = 100   W_time = 15   tau = 30 min"))
    story.append(Spacer(1, 5))
    story += bullets([
        "<b>Tier preservation:</b> each ESI level is worth 1,000 points, while risk adds at most 100 and waiting adds "
        "about 36 points after 5 hours, so a lower-acuity patient can never overtake a higher-acuity one.",
        "<b>Ordering:</b> scores are recalculated on every read and the queue is ranked with a heap (heapq on negated "
        "scores); the top entry is the next patient called to a bed.",
        "<b>Time advance:</b> staff enter any number of minutes to advance all waiting times and re-score the queue.",
        "<b>Overrides and audit:</b> a nurse override sets a new ESI, re-scores and re-sorts immediately, and appends an "
        "audit entry (timestamp, patient, clinician ID, original and new ESI, reason, resulting score).",
    ])

    # ------------------------------------------------------------------
    # 9. INTERFACES
    # ------------------------------------------------------------------
    story.append(Paragraph("9. Application Interfaces", h1))
    story.append(Paragraph("9.1 Streamlit Dashboard (ui/app.py)", h2))
    story.append(Paragraph(
        "Three tabs: <b>Stage 1: Patient Intake &amp; Assessment</b> (case presets, intake form, results with SHAP chart, "
        "rules and grounded narrative), <b>Stage 2: Queue Scheduler</b> (live ranked queue, demo population, custom time "
        "advance, bed assignment, nurse override) and <b>Nurse Override Audit Log</b>. Theme settings live in "
        ".streamlit/config.toml.", body))
    story.append(Paragraph("9.2 FastAPI Service (api/main.py)", h2))
    story.append(table([
        ["Endpoint", "Purpose"],
        ["GET /health", "Service status, loaded models and queue size"],
        ["POST /analyze-patient", "Full Stage 1 assessment without changing the queue"],
        ["GET /queue", "Ranked queue and override count"],
        ["POST /queue/add", "Assess a patient and add them to the queue"],
        ["POST /queue/advance-time", "Advance all wait times by 0.1–120 minutes"],
        ["POST /queue/pop-next", "Call the highest-priority patient to a bed"],
        ["POST /override", "Record a nurse override and re-sort"],
        ["POST /queue/populate-demo", "Fill the queue with 1–50 synthetic patients"],
    ], [140, 364]))

    # ------------------------------------------------------------------
    # 10. VERIFICATION & ENVIRONMENT
    # ------------------------------------------------------------------
    story.append(Paragraph("10. Verification Suite (19 tests)", h1))
    story.append(table([
        ["Test Module", "Scope", "Tests"],
        ["test_rule_engine.py", "Decision Points A–D, frailty guard, resource fallback, min(ML, floor) bounding", "7"],
        ["test_scheduler.py", "Priority formula, ESI tier preservation, intra-tier risk ordering, time advancement, "
                              "queue ranking, deterioration re-sort, override and pop", "5"],
        ["test_grounding.py", "Grounded narratives, inverted-SHAP rejection, floor-breach rejection, numeric fidelity", "4"],
        ["test_api.py", "Health and analysis endpoints; queue add, advance-time, override and pop-next lifecycle", "3"],
    ], [110, 344, 50]))
    story.append(Paragraph("11. Environment &amp; Commands", h1))
    story.append(Paragraph(
        "Python 3.13 with xgboost 3.4, shap 0.52, scikit-learn 1.9, numpy 2.5, pandas 3.0, streamlit 1.62, altair 6.2, "
        "fastapi 0.141, uvicorn 0.52, pydantic 2.13, requests 2.34 and pytest 9.1.", body))
    story.append(table([
        ["Command", "Action"],
        ["python main.py --ui", "Launch the dashboard at http://localhost:8501"],
        ["python main.py --api", "Launch the REST API at http://localhost:8000 (docs at /docs)"],
        ["python main.py --simulate", "Assess 10 synthetic patients in the terminal"],
        ["python main.py --test", "Run the 19-test verification suite"],
        ["python -m erflow.scripts.train_models", "Regenerate the dataset and retrain all three models"],
        ["python -m erflow.scripts.generate_documentation_pdf", "Rebuild this document"],
    ], [200, 304]))
    story.append(Spacer(1, 10))
    story.append(KeepTogether(box(Paragraph(
        "<b>Decision Support Notice:</b> ERflow provides risk estimates, safety floors and grounded explanations to "
        "support clinicians. Final acuity decisions and clinical disposition always remain with the attending healthcare "
        "professional, who can override any recommendation.", callout),
        bg=colors.HexColor("#EFF6FF"), border=colors.HexColor("#93C5FD"))))

    doc.build(story, canvasmaker=DocumentationCanvas)
    print(f"Technical documentation PDF successfully generated at: {output_pdf_path}")


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_file = os.path.join(root_dir, "ERflow_Technical_Documentation.pdf")
    generate_technical_documentation_pdf(out_file)
