"""
Script to generate a dedicated technical documentation PDF explaining:
"How ERflow Handles ESI Score Disagreements, Model Conflicts, and Acuity Mismatches"
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas for dynamic total page numbering and running headers/footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))

        if self._pageNumber > 1:
            # Header
            self.drawString(54, letter[1] - 36, "ERflow — ESI Score Mismatch & Conflict Resolution Architecture")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Clinical Safety & Governance Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

            # Footer
            self.line(54, 45, letter[0] - 54, 45)
            self.drawString(54, 32, "Confidential — Team Argo (Karan Aditya, Jai A Mishra) | IIT Guwahati")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(letter[0] - 54, 32, page_text)

        self.restoreState()


def generate_esi_mismatch_pdf(output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Color definitions
    primary_blue = colors.HexColor("#1E3A8A")
    teal_accent = colors.HexColor("#0D9488")
    dark_slate = colors.HexColor("#1F2937")
    alert_red = colors.HexColor("#DC2626")
    warning_amber = colors.HexColor("#D97706")
    success_green = colors.HexColor("#16A34A")
    light_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=25,
        textColor=primary_blue,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=17,
        textColor=primary_blue,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=dark_slate,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'Code_Block',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=dark_slate
    )

    story = []

    # ==========================================
    # TITLE & HEADER
    # ==========================================
    story.append(Paragraph("ERflow", title_style))
    story.append(Paragraph("ESI Score Disagreement & Conflict Resolution Architecture Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_blue, spaceBefore=0, spaceAfter=8))

    meta_table_data = [
        [
            Paragraph("<b>Project:</b> ERflow — Emergency Department Decision Support Prototype", body_style),
            Paragraph("<b>Team:</b> Argo | IIT Guwahati", body_style)
        ],
        [
            Paragraph("<b>Subject:</b> Multi-Model Conflict Resolution & Safety Floor Guarantees", body_style),
            Paragraph("<b>Document Type:</b> Clinical Safety &amp; Governance Guide", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 234])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Core Problem Callout Box
    problem_summary = (
        "<b>The Clinical Challenge:</b> In an Emergency Department, disagreements can arise between "
        "<b>probabilistic machine learning predictions</b>, <b>deterministic clinical rule engines (ESI v5)</b>, "
        "and <b>human clinician judgments</b>. If an AI predicts standard acuity (ESI 3) for a patient who has dangerous "
        "hidden frailty, or if the AI predicts critical acuity (ESI 2) when standard protocol indicates ESI 4, how does the system resolve "
        "the conflict safely without causing mortality risks (under-prioritization) or emergency department gridlock (over-prioritization)?"
    )
    prob_box = Table([[Paragraph(problem_summary, callout_style)]], colWidths=[504])
    prob_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(prob_box)
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 1: SOURCES OF ESI ASSIGNMENT
    # ==========================================
    story.append(Paragraph("1. The Three Independent Acuity Sources", h1_style))
    story.append(Paragraph(
        "ERflow decouples acuity determination into three distinct evaluation layers. A mismatch occurs whenever these layers produce conflicting acuity levels:",
        body_style
    ))

    sources_data = [
        [
            Paragraph("Evaluation Layer", table_header_style),
            Paragraph("Mechanism & Output", table_header_style),
            Paragraph("Clinical Focus", table_header_style),
            Paragraph("Potential Failure Mode", table_header_style)
        ],
        [
            Paragraph("<b>1. Demographic XGBoost Agent</b><br/>(ML Layer)", table_cell_style),
            Paragraph("Continuous risk probability <code>P_risk</code> (30-day mortality / ICU admission).<br/>Mapped at threshold <code>t = 0.504</code>:<br/>• P_risk ≥ 0.504 → <b>ESI 2</b><br/>• P_risk &lt; 0.504 → <b>ESI 3</b>", table_cell_style),
            Paragraph("Complex non-linear interactions across age, vitals, frailty (CFS), and comorbidities.", table_cell_style),
            Paragraph("May miss rare deterministic red flags (e.g. stroke FAST+, neonatal fever) if individual vitals look normal.", table_cell_style)
        ],
        [
            Paragraph("<b>2. Deterministic ESI v5 Rules</b><br/>(Safety Floor)", table_cell_style),
            Paragraph("Pure Python rule engine evaluating Decision Points A, B, C, D, and Frailty Guard.<br/>Outputs hard minimum acuity floor: <code>ABCDE_ESI_Floor</code> (1 to 5).", table_cell_style),
            Paragraph("Immediate life threats (ESI 1), acute high-risk situations (ESI 2), and vital danger zones (ESI 2).", table_cell_style),
            Paragraph("Cannot quantify multi-variable holistic decompensation risks when all individual vitals are borderline.", table_cell_style)
        ],
        [
            Paragraph("<b>3. Bedside Nurse</b><br/>(Human-in-the-Loop)", table_cell_style),
            Paragraph("Bedside clinical judgment, visual patient inspection, and intuition.<br/>Can confirm or input manual <code>Nurse_Override_ESI</code>.", table_cell_style),
            Paragraph("Physical appearance, diaphoresis, pallor, subtle frailty, unquantified distress.", table_cell_style),
            Paragraph("Cognitive fatigue during high-volume shifts, subjective inconsistency.", table_cell_style)
        ]
    ]
    sources_table = Table(sources_data, colWidths=[95, 145, 135, 129])
    sources_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_blue),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(sources_table)

    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 2: THE 4 DISAGREEMENT SCENARIOS
    # ==========================================
    story.append(Paragraph("2. The Four Disagreement Scenarios & Resolution Mechanics", h1_style))
    story.append(Paragraph(
        "ERflow enforces strict mathematical and clinical rules to resolve conflicts across every possible permutation:",
        body_style
    ))

    mismatch_data = [
        [
            Paragraph("Mismatch Scenario", table_header_style),
            Paragraph("Condition & Clinical Example", table_header_style),
            Paragraph("System Resolution Mechanism", table_header_style),
            Paragraph("Resulting Final ESI", table_header_style)
        ],
        [
            Paragraph("<b>Case 1: Rule Safety Floor Escalation</b><br/>(ML = 3, Floor = 1 or 2)", table_cell_style),
            Paragraph("<b>Example:</b> 71y stroke patient with normal heart rate (78 bpm) and SpO2 (98%). ML predicts ESI 3 (low mortality risk), but Decision Point B detects FAST+ facial droop.", table_cell_style),
            Paragraph("<b>Deterministic Safety Floor Override:</b><br/><code>Final_ESI = min(ML_ESI, Rule_Floor)</code><br/>min(3, 2) = 2. The deterministic safety rule unconditionally overrides ML. <b>AI is mathematically forbidden from downgrading.</b>", table_cell_style),
            Paragraph("<font color='#DC2626'><b>ESI LEVEL 2</b></font><br/>(Rule Escalation / Hard Locked)", table_cell_style)
        ],
        [
            Paragraph("<b>Case 2: ML Predictive Risk Escalation</b><br/>(ML = 2, Floor = 3 or 4)", table_cell_style),
            Paragraph("<b>Example:</b> 41y adult with borderline vitals (HR 98, RR 19) and 1 expected resource. ESI v5 rules evaluate ESI 4. However, XGBoost detects multi-variable interaction yielding P_risk = 90.7% ≥ 0.504.", table_cell_style),
            Paragraph("<b>Predictive Risk Escalation:</b><br/><code>Final_ESI = min(ML_ESI, Rule_Floor)</code><br/>min(2, 4) = 2. Acuity is upgraded to ESI 2. UI flags <code>ML_PREDICTIVE_ESCALATION</code> and displays SHAP waterfall charts explaining risk drivers.", table_cell_style),
            Paragraph("<font color='#EA580C'><b>ESI LEVEL 2</b></font><br/>(ML Predictive Escalation)", table_cell_style)
        ],
        [
            Paragraph("<b>Case 3: Epistemic Uncertainty Mismatch</b><br/>(Tree Confidence &lt; 20%)", table_cell_style),
            Paragraph("<b>Example:</b> Patient arrives with conflicting vital patterns or unverified zero-history records. Boosting trees produce high variance (std dev &gt; 0.20) across sub-stages.", table_cell_style),
            Paragraph("<b>Mandatory Human Review Gate:</b><br/>Confidence mapped to &lt;20%. System automatically locks record, flags <code>requires_human_review = True</code>, and triggers prominent clinician alert banner in UI.", table_cell_style),
            Paragraph("<font color='#D97706'><b>MANDATORY NURSE CHECKPOINT</b></font><br/>(AI Flags Model Confusion)", table_cell_style)
        ],
        [
            Paragraph("<b>Case 4: Clinician Manual Override</b><br/>(Nurse disagrees with AI)", table_cell_style),
            Paragraph("<b>Example:</b> AI assesses patient as ESI 3, but the bedside nurse observes acute diaphoresis, pallor, or subtle high-risk frailty not captured in intake fields.", table_cell_style),
            Paragraph("<b>Clinician Override & Instant Re-Sort:</b><br/>Nurse inputs override ESI and mandatory clinical reason. System updates record, re-indexes Max Heap in O(log N) time, and logs an immutable audit trail entry.", table_cell_style),
            Paragraph("<font color='#1E3A8A'><b>NURSE OVERRIDE ESI</b></font><br/>(Clinician Retains Full Autonomy)", table_cell_style)
        ]
    ]
    mismatch_table = Table(mismatch_data, colWidths=[105, 140, 165, 94])
    mismatch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(mismatch_table)

    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 3: MATHEMATICAL BOUNDING FORMULA
    # ==========================================
    story.append(Paragraph("3. The Asymmetric Safety Bounding Rule", h1_style))
    story.append(Paragraph(
        "At the core of the conflict resolution architecture is the <b>Asymmetric Safety Bounding Rule</b>. "
        "Because lower ESI numbers represent higher clinical urgency (ESI 1 = highest, ESI 5 = lowest), the mathematical <code>min()</code> function creates a one-way safety ratchet:",
        body_style
    ))

    formula_box_text = (
        "<b>THE SAFETY RATCHET EQUATION:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Final_ESI = min(ML_ESI_Recommendation, ABCDE_ESI_Floor)</b><br/><br/>"
        "<b>Mathematical Consequences:</b><br/>"
        "• <b>Priority Can Only Be Escalated:</b> If either the ML model OR the deterministic rule engine identifies a danger signal, the patient is escalated to the highest severity level.<br/>"
        "• <b>Downgrading is Impossible:</b> An ML model with low predicted mortality CANNOT downgrade a patient whose vital signs or symptoms trigger an ESI 1 or ESI 2 deterministic floor.<br/>"
        "• <b>Zero-History Safety:</b> If intake data is missing (e.g. resources uncollected), Decision Point C gracefully stubs as <code>insufficient_data</code>, preventing false ESI 5 downgrades."
    )
    formula_box = Table([[Paragraph(formula_box_text, code_style)]], colWidths=[504])
    formula_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(formula_box)

    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 4: HOW SCHEDULER HANDLES MISMATCHES
    # ==========================================
    story.append(Paragraph("4. Impact of Score Disagreements on the Dynamic Scheduler", h1_style))
    story.append(Paragraph(
        "When an ESI score is escalated or overridden, Stage 2 (the Continuous Max Heap scheduler) immediately updates the patient's queue position using the dynamic priority formula:",
        body_style
    ))

    sched_text = (
        "<b>Dynamic Priority Score Equation:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>Priority Score(t) = W_floor × (6 - ESI_final) + W_risk × P_risk + W_time × ln(1 + t_wait / 30)</code><br/><br/>"
        "• <b>Massive Score Boost on Escalation:</b> Because <code>W_floor = 1000</code>, escalating an ESI 3 to ESI 2 increases the base score from 3,000 to <b>4,000 points (+1,000 pts)</b>, instantly propelling the patient to the top of the waiting queue.<br/>"
        "• <b>Preserving Clinical Safety Tiers:</b> The waiting-time bonus grows only logarithmically: even after 5 hours of waiting it adds roughly 36 points. This mathematically prevents any lower-acuity patient (ESI 4 base = 2,000) from ever jumping ahead of an escalated high-acuity patient (ESI 2 base = 4,000).<br/>"
        "• <b>Intra-Tier Resolution:</b> If two patients have the same ESI 2 level, the continuous <code>P_risk</code> (weighted by <code>W_risk = 100</code>) breaks the tie (e.g. 90% risk gets +90 pts vs. 15% risk gets +15 pts)."
    )
    sched_box = Table([[Paragraph(sched_text, body_style)]], colWidths=[504])
    sched_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sched_box)

    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 5: GROUNDING & ANTI-HALLUCINATION
    # ==========================================
    story.append(Paragraph("5. Anti-Hallucination Grounding Validator During Disagreements", h1_style))
    story.append(Paragraph(
        "When disagreements occur, the local LLM reasoning synthesizer (or deterministic engine) must generate a clinical explanation. "
        "The <b>Grounding Validator</b> (<code>explain/grounding.py</code>) verifies that the generated explanation never hallucinates:",
        body_style
    ))
    story.append(Paragraph("• <b>Acuity Floor Integrity Check:</b> Rejects any explanation where the recommended ESI is less severe than the deterministic safety floor (e.g. claiming ESI 4 when floor is ESI 2).", bullet_style))
    story.append(Paragraph("• <b>Directional SHAP Consistency Check:</b> Rejects any claim stating a feature 'increased risk' if its true SHAP attribution is negative (protective).", bullet_style))
    story.append(Paragraph("• <b>Numerical Fidelity Check:</b> Ensures cited vitals in the narrative match raw patient intake records within 0.5 units.", bullet_style))

    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 6: AUDIT TRAIL & GOVERNANCE
    # ==========================================
    story.append(Paragraph("6. Clinical Governance &amp; Audit Trail", h1_style))
    story.append(Paragraph(
        "Every conflict resolution and clinician override is permanently recorded into an auditable log:",
        body_style
    ))

    audit_example_text = (
        "<b>Audit Trail Log Structure:</b><br/>"
        "• <b>Timestamp:</b> <code>1756468200.12</code> | <b>Patient ID:</b> <code>PID_00005</code><br/>"
        "• <b>Original ML Recommendation:</b> ESI 3 (P_risk: 28.8%, Confidence: 68.4%)<br/>"
        "• <b>Triggered Rule:</b> <code>Decision Point D: High-Risk Tachycardia (HR 108 > 100 bpm)</code><br/>"
        "• <b>Automated Resolution:</b> <code>RULE_SAFETY_ESCALATION → Final Acuity: ESI 2</code><br/>"
        "• <b>Clinician Override (if any):</b> Recorded with Clinician ID, reason, and resulting Max Heap re-sort."
    )
    audit_box = Table([[Paragraph(audit_example_text, code_style)]], colWidths=[504])
    audit_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(audit_box)

    story.append(Spacer(1, 10))

    # Summary Callout
    summary_footer = (
        "<b>Summary:</b> ERflow solves ESI score disagreements through mathematical safety floors, "
        "epistemic tree variance uncertainty gates, real-time Max Heap priority re-sorting, and unrestricted clinician override authority—"
        "ensuring patient safety is never compromised by black-box algorithms."
    )
    summary_footer_box = Table([[Paragraph(summary_footer, callout_style)]], colWidths=[504])
    summary_footer_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_footer_box)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"ESI Disagreement PDF successfully generated at: {output_pdf_path}")


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_file = os.path.join(root_dir, "ERflow_ESI_Disagreement_and_Conflict_Resolution.pdf")
    generate_esi_mismatch_pdf(out_file)

