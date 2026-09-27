"""
ERflow - Streamlit Interactive Clinical Decision Support & Dynamic Scheduler Dashboard
Team: Argo (Karan Aditya, Jai A Mishra) - IIT Guwahati
"""

import os
import json
import html
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

from erflow.agent.orchestrator import AssessmentOrchestrator
from erflow.scheduler.max_heap import DynamicMaxHeapQueue
from erflow.scripts.generate_data import generate_synthetic_ed_data

# Page configuration
st.set_page_config(
    page_title="ERflow | Emergency Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling (warm hospital theme; base colours and fonts live in .streamlit/config.toml)
st.markdown("""
<style>
    :root {
        --sa-green: #3E7350;
        --sa-green-dark: #336142;
        --sa-terracotta: #9f3c16;
        --sa-surface: #fff8f3;
        --sa-surface-low: #fcf2e8;
        --sa-surface-mid: #f7ece2;
        --sa-surface-high: #f1e6dc;
        --sa-card: #ffffff;
        --sa-text: #1f1b15;
        --sa-text-muted: #57423b;
        --sa-outline: #ebe1d7;
        --sa-error: #ba1a1a;
        --sa-tertiary: #346645;
        --sa-heading: 'Plus Jakarta Sans', 'Inter', sans-serif;
        --sa-body: 'Inter', sans-serif;
    }
    .block-container { padding-top: 1.25rem; padding-bottom: 3rem; padding-left: 3rem; padding-right: 3rem; max-width: 1600px; }
    [data-testid="stHeader"] { background: transparent; }

    /* Material icon glyphs in custom HTML (font bundled with Streamlit) */
    .msi {
        font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
        font-size: 18px; line-height: 1; letter-spacing: normal; text-transform: none;
        display: inline-block; white-space: nowrap; direction: ltr; vertical-align: middle;
        -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga';
    }

    /* Brand bar */
    .sa-brand { display: flex; align-items: center; justify-content: center; gap: 18px; padding: 10px 0 22px 0; }
    .sa-logo { width: 72px; height: 72px; border-radius: 17px; background: #bf542c; display: flex;
               align-items: center; justify-content: center; box-shadow: 0 3px 10px rgba(159,60,22,0.28); }
    .sa-brand-title { font-family: var(--sa-heading); font-size: 46px; font-weight: 700; color: var(--sa-text);
                      letter-spacing: -0.02em; line-height: 1.1; }
    .sa-chip { display: inline-block; padding: 2px 8px; border-radius: 6px; background: var(--sa-surface-high);
               color: var(--sa-text-muted); font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
               text-transform: uppercase; vertical-align: middle; }
    .sa-chip-green { background: #dcefe0; color: #1e5031; }
    .sa-chip-red { background: #ffdad6; color: #93000a; }
    .sa-chip-amber { background: #ffdcbd; color: #693c00; }
    .sa-chip-terra { background: #ffdbcf; color: #822801; }

    /* Tabs styled as pill navigation */
    .stTabs [role="tablist"] { gap: 6px; background: #efe2d4; padding: 6px; border-radius: 14px;
                               width: fit-content; max-width: 100%; border: none; box-shadow: none;
                               margin-left: auto; margin-right: auto; }
    .stTabs [data-testid="stTab"] { height: auto; padding: 15px 20px; border-radius: 10px; background: transparent;
                                    color: var(--sa-text-muted); white-space: nowrap; }
    /* Icon and label as a centred flex row so both sit exactly mid-height in the pill */
    .stTabs [data-testid="stTab"] p { font-size: 15px; font-weight: 500; margin: 0; line-height: 1;
                                      display: flex; align-items: center; gap: 7px; }
    .stTabs [data-testid="stTab"] p span[role="img"] { font-size: 19px !important; line-height: 1; vertical-align: 0; margin: 0; }
    .stTabs [data-testid="stTab"][aria-selected="true"] { background: var(--sa-green) !important; color: #ffffff !important;
                                                          box-shadow: 0 1px 4px rgba(46, 86, 59, 0.25); }
    .stTabs [data-testid="stTab"]:not([aria-selected="true"]):hover { background: #e6d6c5; }
    .stTabs [data-testid="stTab"][aria-selected="true"] p,
    .stTabs [data-testid="stTab"][aria-selected="true"] span { color: #ffffff !important; font-weight: 600; }
    .stTabs [role="tabpanel"] { padding-top: 22px; }
    .stTabs .react-aria-SelectionIndicator, .stTabs [role="tablist"]::after { display: none; }
    @media (max-width: 1180px) {
        .stTabs [data-testid="stTab"] { padding: 13px 14px; }
        .stTabs [data-testid="stTab"] p { font-size: 14px; }
        .stTabs [data-testid="stTab"] p span[role="img"] { font-size: 17px !important; }
    }

    /* Section headings */
    .sa-eyebrow { display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600;
                  letter-spacing: 0.06em; text-transform: uppercase; color: var(--sa-terracotta); }
    .sa-eyebrow .sa-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--sa-terracotta); }
    .sa-eyebrow .sa-eyebrow-muted { color: var(--sa-text-muted); text-transform: none; letter-spacing: 0.02em; }
    .sa-h1 { font-family: var(--sa-heading); font-size: 26px; font-weight: 700; color: var(--sa-text);
             letter-spacing: -0.015em; margin: 4px 0 14px 0; line-height: 1.25; }
    .sa-section-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; flex-wrap: wrap; }

    /* Cards (keyed Streamlit containers) */
    [class*="st-key-card_"] { background: var(--sa-card); border-radius: 12px; padding: 20px 22px;
                              box-shadow: 0 4px 20px rgba(46, 50, 48, 0.06), 0 1px 2px rgba(74, 60, 49, 0.05); }
    .st-key-presetbar { background: var(--sa-surface-low); border-radius: 12px; padding: 8px 10px; }
    .sa-card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px;
                    background: var(--sa-surface-low); border-radius: 8px; padding: 8px 12px; margin-bottom: 4px; }
    .sa-card-head .sa-card-title { display: flex; align-items: center; gap: 8px; font-family: var(--sa-heading);
                                   font-size: 16px; font-weight: 600; color: var(--sa-text); }
    .sa-card-head .msi { color: var(--sa-terracotta); }
    .sa-card-tag { font-size: 11px; font-weight: 600; color: var(--sa-text-muted); letter-spacing: 0.03em; }
    .sa-card-plain-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 4px; }
    .sa-card-plain-head .sa-card-title { display: flex; align-items: center; gap: 10px; font-family: var(--sa-heading);
                                         font-size: 18px; font-weight: 600; color: var(--sa-text); }
    .sa-card-plain-head .msi { color: var(--sa-terracotta); background: var(--sa-surface-low); border-radius: 8px; padding: 6px; }
    .sa-muted { color: var(--sa-text-muted); font-size: 12px; }
    .sa-footnote { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; background: var(--sa-surface-mid);
                   border-radius: 8px; padding: 8px 12px; font-size: 12px; color: var(--sa-text-muted); margin-top: 8px; }
    .sa-footnote b { color: var(--sa-text); }
    .sa-subhead { font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
                  color: var(--sa-text-muted); margin: 6px 0 2px 0; }

    /* Widget labels: small uppercase captions like the wireframe */
    [data-testid="stWidgetLabel"] p { font-size: 11px !important; font-weight: 600; letter-spacing: 0.05em;
                                      text-transform: uppercase; color: var(--sa-text-muted); }
    [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label p { font-size: 13px !important; font-weight: 400; letter-spacing: 0;
                                         text-transform: none; color: var(--sa-text); }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button { font-weight: 600; min-height: 42px; }
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
        background: var(--sa-green); border: 1px solid var(--sa-green-dark);
        box-shadow: 0 2px 6px rgba(46, 86, 59, 0.25); }
    .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
        background: var(--sa-green-dark); }
    .stButton > button[kind="secondary"], .stFormSubmitButton > button[kind="secondaryFormSubmit"] {
        background: var(--sa-surface-high); border: 1px solid transparent; color: var(--sa-text); }
    .stButton > button[kind="secondary"]:hover { border-color: var(--sa-green); color: var(--sa-green); }
    .st-key-presetbar .stButton > button { height: 42px; min-height: 42px; max-height: 42px; padding: 4px 11px;
                                           white-space: nowrap; }
    .st-key-presetbar .stButton > button[kind="secondary"] { background: #ffffff;
        box-shadow: 0 1px 2px rgba(74, 60, 49, 0.08); }
    .st-key-presetbar .stButton > button p { font-size: 13px; font-weight: 500; line-height: 1.25; white-space: nowrap; }
    .st-key-preset_cardiac_arrest [data-testid="stIconMaterial"] { color: #ba1a1a; }
    .st-key-preset_stroke [data-testid="stIconMaterial"] { color: #bf542c; }
    .st-key-preset_frail_elderly [data-testid="stIconMaterial"] { color: #88520c; }
    .st-key-preset_pediatric_fever [data-testid="stIconMaterial"] { color: #fdb569; }
    .st-key-preset_borderline_adult [data-testid="stIconMaterial"] { color: #ffb86e; }
    .st-key-preset_minor_sprain [data-testid="stIconMaterial"] { color: #346645; }
    .st-key-presetbar .stButton > button[kind="primary"] [data-testid="stIconMaterial"] { color: #ffdbcf; }
    .st-key-presetbar [data-testid="stIconMaterial"] { font-variation-settings: 'FILL' 1; font-size: 12px; }
    .sa-preset-label { font-size: 11px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
                       color: var(--sa-text-muted); padding: 0 4px; white-space: nowrap; line-height: 42px; }
    .st-key-presetbar [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    /* Narrower screens: label gets its own line so all six presets still fit on one row */
    @media (max-width: 1400px) {
        .st-key-presetbar [data-testid="stElementContainer"]:has(.sa-preset-label) { flex-basis: 100%; }
        .st-key-presetbar .sa-preset-label { line-height: 20px; }
    }

    /* Metric tiles */
    .sa-metrics { display: grid; gap: 14px; margin: 6px 0 16px 0; }
    .sa-metrics-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
    .sa-metrics-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    @media (max-width: 1100px) { .sa-metrics-5 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 700px) { .sa-metrics-5, .sa-metrics-3 { grid-template-columns: 1fr; } }
    .sa-tile { background: var(--sa-card); border-radius: 12px; padding: 14px 16px; display: flex; flex-direction: column;
               justify-content: space-between; gap: 8px; min-height: 118px;
               box-shadow: 0 4px 20px rgba(46, 50, 48, 0.06), 0 1px 2px rgba(74, 60, 49, 0.05); }
    .sa-tile-accent { border-left: 4px solid var(--sa-terracotta); }
    .sa-tile-label { font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--sa-text-muted);
                     display: flex; justify-content: space-between; align-items: center; gap: 6px; }
    .sa-tile-value { font-family: var(--sa-body); font-size: 28px; font-weight: 700; letter-spacing: -0.02em;
                     color: var(--sa-text); line-height: 1.1; }
    .sa-tile-value small { font-size: 13px; font-weight: 500; color: var(--sa-text-muted); letter-spacing: 0; }
    .sa-tile-foot { font-size: 12px; font-weight: 500; color: var(--sa-text-muted); display: flex; align-items: center; gap: 4px; }
    .sa-tile-foot .msi { font-size: 15px; }
    .sa-esi-solid { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 8px;
                    color: #ffffff; font-size: 18px; font-weight: 700; width: fit-content; }
    .sa-consensus { display: flex; align-items: center; gap: 8px; border-radius: 8px; padding: 8px 10px;
                    font-size: 13px; font-weight: 600; }

    /* Rule list + narrative */
    .sa-rule { display: flex; gap: 10px; align-items: flex-start; background: var(--sa-surface-low); border-radius: 10px;
               padding: 10px 12px; margin: 8px 0; font-size: 13px; color: var(--sa-text); line-height: 1.5; }
    .sa-rule .msi { color: var(--sa-terracotta); font-size: 18px; margin-top: 1px; }
    .sa-narrative { background: var(--sa-surface-low); border-radius: 10px; padding: 14px 16px; font-size: 14px;
                    line-height: 1.7; color: var(--sa-text); margin: 8px 0 12px 0; }
    .sa-callout { display: flex; gap: 10px; align-items: flex-start; border-radius: 10px; padding: 12px 14px; font-size: 13px;
                  line-height: 1.55; margin: 6px 0; }
    .sa-callout .msi { font-size: 20px; }
    .sa-callout-ok { background: #dcefe0; color: #1e5031; }
    .sa-callout-bad { background: #ffdad6; color: #93000a; }
    .sa-callout-note { background: var(--sa-surface-low); color: var(--sa-text-muted); }
    .sa-callout-note .msi { color: var(--sa-terracotta); }

    /* Stage 2 controls: uniform control height */
    .st-key-card_controls .stButton > button, .st-key-card_controls .stFormSubmitButton > button {
        height: 48px; min-height: 48px; max-height: 48px; padding: 4px 12px; }
    .st-key-card_controls .stButton > button p, .st-key-card_controls .stFormSubmitButton > button p {
        font-size: 13px; line-height: 1.2; }
    .st-key-card_controls [data-testid="stTextInputRootElement"] { height: 48px; }
    .st-key-card_controls [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    .st-key-card_controls [data-testid="stForm"] { padding: 0; margin: 0; }


    /* Tables */
    .sa-table-wrap { overflow-x: auto; border-radius: 12px; margin-top: 8px; }
    table.sa-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    table.sa-table thead th { background: var(--sa-surface-high); color: var(--sa-text-muted); font-size: 11px; font-weight: 600;
                              letter-spacing: 0.05em; text-transform: uppercase; text-align: left; padding: 11px 12px; white-space: nowrap; }
    table.sa-table tbody td { padding: 11px 12px; border-bottom: 1px solid var(--sa-outline); color: var(--sa-text); vertical-align: middle; }
    table.sa-table tbody tr:last-child td { border-bottom: none; }
    table.sa-table tbody tr.sa-root td { background: var(--sa-surface-low); }
    @keyframes sa-new-flash { 0% { background: #b9e2c4; } 100% { background: #e3f2e7; } }
    table.sa-table tbody tr.sa-new td { background: #e3f2e7; animation: sa-new-flash 2.5s ease-out; }
    table.sa-table tbody tr.sa-new td:first-child { box-shadow: inset 4px 0 0 var(--sa-green); }
    .sa-new-tag { display: inline-block; margin-left: 6px; padding: 1px 6px; border-radius: 4px; background: #bf542c;
                  color: #ffffff; font-size: 10px; font-weight: 700; letter-spacing: 0.06em; }
    .sa-rank { display: inline-flex; align-items: center; justify-content: center; min-width: 30px; height: 30px; border-radius: 50%;
               font-weight: 700; font-size: 12px; }
    .sa-rank-root { background: var(--sa-green); color: #ffffff; }
    .sa-root-tag { display: inline-block; margin-left: 6px; padding: 1px 6px; border-radius: 4px; background: var(--sa-green);
                   color: #ffffff; font-size: 10px; font-weight: 700; letter-spacing: 0.06em; }
    .sa-esi-soft { display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; white-space: nowrap; }
    .sa-table-foot { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; background: var(--sa-surface-mid);
                     padding: 8px 12px; font-size: 12px; color: var(--sa-text-muted); border-radius: 0 0 12px 12px; }
    .sa-strong-terra { color: var(--sa-terracotta); font-weight: 700; }

    /* Bed assignment */
    .sa-candidate { background: var(--sa-surface-low); border-radius: 10px; padding: 14px 16px; margin: 8px 0 12px 0; }
    .sa-candidate-top { display: flex; justify-content: space-between; font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
                        text-transform: uppercase; color: var(--sa-terracotta); }
    .sa-candidate-name { font-family: var(--sa-heading); font-size: 18px; font-weight: 600; color: var(--sa-text); margin: 6px 0; }
    .sa-candidate-meta { font-size: 13px; color: var(--sa-text-muted); display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

</style>
""", unsafe_allow_html=True)


# --- Presentation helpers (HTML rendering only; no clinical logic) ---
ESI_STYLES = {
    1: {"solid": "#ba1a1a", "soft_bg": "#ffdad6", "soft_fg": "#93000a", "name": "Resuscitation"},
    2: {"solid": "#9f3c16", "soft_bg": "#ffdbcf", "soft_fg": "#822801", "name": "Emergent"},
    3: {"solid": "#88520c", "soft_bg": "#ffdcbd", "soft_fg": "#693c00", "name": "Urgent"},
    4: {"solid": "#346645", "soft_bg": "#dcefe0", "soft_fg": "#1e5031", "name": "Less Urgent"},
    5: {"solid": "#4a6572", "soft_bg": "#e6edf0", "soft_fg": "#23404d", "name": "Non-Urgent"},
}


def esc(value) -> str:
    return html.escape(str(value))


def icon(name: str, style: str = "") -> str:
    return f'<span class="msi" style="{style}">{name}</span>'


def render_html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def section_heading(eyebrow: str, eyebrow_muted: str, title: str, right_html: str = ""):
    render_html(
        f'<div class="sa-section-row"><div>'
        f'<div class="sa-eyebrow">{esc(eyebrow)}<span class="sa-dot"></span>'
        f'<span class="sa-eyebrow-muted">{esc(eyebrow_muted)}</span></div>'
        f'<div class="sa-h1">{esc(title)}</div></div>'
        f'<div style="padding-bottom:14px">{right_html}</div></div>'
    )


def card_head(icon_name: str, title: str, tag: str = ""):
    render_html(
        f'<div class="sa-card-head"><div class="sa-card-title">{icon(icon_name)}{esc(title)}</div>'
        f'<span class="sa-card-tag">{esc(tag)}</span></div>'
    )


def card_plain_head(icon_name: str, title: str, subtitle: str = "", right_html: str = ""):
    sub = f'<div class="sa-muted">{esc(subtitle)}</div>' if subtitle else ""
    render_html(
        f'<div class="sa-card-plain-head"><div class="sa-card-title">{icon(icon_name)}<div>{esc(title)}{sub}</div></div>'
        f'<div>{right_html}</div></div>'
    )


def esi_soft_badge(esi: int) -> str:
    s = ESI_STYLES.get(int(esi), ESI_STYLES[3])
    return (f'<span class="sa-esi-soft" style="background:{s["soft_bg"]};color:{s["soft_fg"]}">'
            f'ESI {int(esi)} ({s["name"]})</span>')


# Initialize Session State
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = AssessmentOrchestrator()

if 'scheduler' not in st.session_state:
    st.session_state.scheduler = DynamicMaxHeapQueue()
    # Populate initial queue with 8 realistic patients
    df_init = generate_synthetic_ed_data(n_samples=8, random_seed=42)
    for _, row in df_init.iterrows():
        p = row.to_dict()
        # Queue rows show no prose, so skip the LLM: 8 patients x ~60s of local
        # inference would leave the first page render blank for minutes.
        res = st.session_state.orchestrator.analyze_patient(
            patient_data=p,
            wait_time_mins=float(p.get('current_wait_time_mins', 10.0)),
            use_llm=False
        )
        st.session_state.scheduler.add_patient(
            patient_id=res['patient_id'],
            esi_final=res['final_esi'],
            p_risk=res['p_risk'],
            wait_time_mins=res['wait_time_mins'],
            age=int(p.get('age', 35)),
            age_cohort=res['ml_details'].get('cohort', 'adult'),
            gender=p.get('gender', 'Unknown'),
            chief_complaint=f"Clinical Presentation ({p.get('age_cohort', 'adult').capitalize()})",
            has_prior_history=int(p.get('has_prior_history', 0)),
            confidence_score=res['confidence_score'],
            requires_human_review=res['requires_human_review'],
            vital_signs={
                'heart_rate': p.get('heart_rate'),
                'resp_rate': p.get('resp_rate'),
                'spo2': p.get('spo2'),
                'sbp': p.get('sbp'),
                'temp_c': p.get('temp_c'),
                'cfs_frailty_score': p.get('cfs_frailty_score')
            }
        )

if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None


# --- Hide Sidebar ---
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
render_html(
    '<div class="sa-brand">'
    '<div class="sa-logo"><svg width="38" height="38" viewBox="0 0 24 24" fill="#ffffff">'
    '<path d="M9 2h6v7h7v6h-7v7H9v-7H2V9h7z"/></svg></div>'
    '<div><div class="sa-brand-title">ERflow</div>'
    '</div></div>'
)

tab1, tab2, tab3 = st.tabs([
    ":material/clinical_notes: Stage 1: Patient Intake & Assessment",
    ":material/account_tree: Stage 2: Queue Scheduler",
    ":material/fact_check: Nurse Override Audit Log"
])


# =========================================================================
# TAB 1: PATIENT INTAKE & ASSESSMENT
# =========================================================================
PRESETS = [
    ("cardiac_arrest", "Cardiac Arrest (ESI 1)"),
    ("stroke", "Stroke / STEMI (ESI 2)"),
    ("frail_elderly", "Frail Elderly (CFS 7)"),
    ("pediatric_fever", "Febrile Infant (Ped)"),
    ("borderline_adult", "Borderline Adult (ESI 3)"),
    ("minor_sprain", "Simple Ankle Sprain (ESI 4)"),
]


def _select_preset(preset_key: str):
    # Persist the selected preset and give the new patient a fresh ID; the form
    # keeps this patient until the analysis button is pressed
    st.session_state.selected_preset = preset_key
    st.session_state.intake_patient_id = f"PID_{np.random.randint(10000, 99999)}"


with tab1:
    section_heading("Protocol Active", "Clinical Intake Mode",
                    "Clinical Intake & Physiological Vitals Assessment")

    # Preset scenarios for quick demo testing
    # One flexible row: each button is as wide as its one-line label, and whole buttons
    # wrap to a new line on narrow screens instead of the label breaking
    with st.container(key="presetbar", horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center", gap="xsmall"):
        st.markdown('<div class="sa-preset-label">Quick Case Presets:</div>', unsafe_allow_html=True, width="content")
        active_preset = st.session_state.get('selected_preset')
        for preset_key, preset_label in PRESETS:
            st.button(
                preset_label,
                key=f"preset_{preset_key}",
                icon=":material/fiber_manual_record:",
                type="primary" if active_preset == preset_key else "secondary",
                on_click=_select_preset,
                args=(preset_key,),
                width="content"
            )

    if 'intake_patient_id' not in st.session_state:
        st.session_state.intake_patient_id = f"PID_{np.random.randint(10000, 99999)}"
    preset_chosen = st.session_state.get('selected_preset')

    # Default form values
    def_age = 45
    def_cohort = "adult"
    def_gender = "Female"
    def_cfs = 2
    def_history = 1
    def_comorb = 1
    def_hr = 82.0
    def_rr = 16.0
    def_spo2 = 98.0
    def_sbp = 125.0
    def_temp = 37.1
    def_complaint = "Acute abdominal pain and nausea"
    def_req_life = 0
    def_high_risk = 0
    def_ams = 0
    def_severe_pain = 0
    def_chest_pain = 0
    def_stroke = 0
    def_resources = 2

    if preset_chosen == "cardiac_arrest":
        def_age = 62
        def_cohort = "adult"
        def_hr = 0.0
        def_rr = 0.0
        def_spo2 = 65.0
        def_sbp = 40.0
        def_complaint = "Unresponsive, apneic, pulseless collapse"
        def_req_life = 1
    elif preset_chosen == "stroke":
        def_age = 71
        def_cohort = "geriatric"
        def_cfs = 4
        def_hr = 104.0
        def_rr = 22.0
        def_spo2 = 93.0
        def_sbp = 185.0
        def_complaint = "Sudden right-sided facial droop and arm weakness (FAST+)"
        def_stroke = 1
        def_high_risk = 1
    elif preset_chosen == "frail_elderly":
        def_age = 84
        def_cohort = "geriatric"
        def_cfs = 7
        def_history = 0  # Zero history test!
        def_comorb = 0
        def_hr = 96.0  # >90 triggers Frailty Guard!
        def_rr = 19.0
        def_spo2 = 94.0
        def_sbp = 115.0
        def_temp = 37.8
        def_complaint = "Generalized weakness, poor oral intake, borderline lethargy"
    elif preset_chosen == "pediatric_fever":
        def_age = 3
        def_cohort = "pediatric"
        def_hr = 148.0  # >140 for toddler triggers Ped Danger Zone!
        def_rr = 38.0   # >35 for toddler triggers Ped Danger Zone!
        def_spo2 = 91.0  # <92% hypoxia
        def_sbp = 82.0
        def_temp = 39.4
        def_complaint = "High fever, rapid grunting respirations, decreased activity"
    elif preset_chosen == "borderline_adult":
        def_age = 34
        def_cohort = "adult"
        def_hr = 88.0
        def_rr = 18.0
        def_spo2 = 97.0
        def_sbp = 120.0
        def_temp = 37.5
        def_complaint = "Moderate abdominal cramps, requires lab work and CT scan"
        def_resources = 2
    elif preset_chosen == "minor_sprain":
        def_age = 24
        def_cohort = "adult"
        def_hr = 72.0
        def_rr = 14.0
        def_spo2 = 99.0
        def_sbp = 118.0
        def_temp = 36.8
        def_complaint = "Twisted ankle while jogging, ambulatory with mild swelling"
        def_resources = 1

    with st.form("patient_intake_form", border=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(key="card_demographics"):
                card_head("badge", "Demographics & History", "SEC-01")
                p_id = st.text_input("Patient Identifier", value=st.session_state.intake_patient_id)
                age = st.number_input("Age (Years)", min_value=0.1, max_value=115.0, value=float(def_age), step=1.0)

                # Cohort auto-selection based on age
                if age < 18:
                    cohort_default_idx = 0
                elif age >= 65:
                    cohort_default_idx = 2
                else:
                    cohort_default_idx = 1

                cohort = st.selectbox("Demographic Age Cohort", ["pediatric", "adult", "geriatric"], index=cohort_default_idx)
                gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0 if def_gender=="Female" else 1)

                has_history = st.radio("Intake History Status", [1, 0], format_func=lambda x: "Verified Records Present" if x==1 else "Zero-History / Unverified Baseline (50% ED Case)", index=0 if def_history==1 else 1)
                comorbidity = st.number_input("Comorbidity Count", min_value=0, max_value=10, value=int(def_comorb))

        with col2:
            with st.container(key="card_vitals"):
                card_head("ecg_heart", "Physiological Vital Signs", "SEC-02")
                hr = st.number_input("Heart Rate (BPM)", min_value=0.0, max_value=250.0, value=float(def_hr), step=1.0)
                rr = st.number_input("Respiratory Rate (Breaths/min)", min_value=0.0, max_value=80.0, value=float(def_rr), step=1.0)
                spo2 = st.number_input("Oxygen Saturation SpO2 (%)", min_value=40.0, max_value=100.0, value=float(def_spo2), step=0.5)
                sbp = st.number_input("Systolic Blood Pressure SBP (mmHg)", min_value=30.0, max_value=260.0, value=float(def_sbp), step=1.0)
                vt_col1, vt_col2 = st.columns(2)
                with vt_col1:
                    temp_c = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=44.0, value=float(def_temp), step=0.1)
                with vt_col2:
                    wait_time = st.number_input("Current Elapsed Wait Time (Mins)", min_value=0.0, max_value=600.0, value=15.0, step=5.0)

        with col3:
            with st.container(key="card_redflags"):
                card_head("clinical_notes", "ESI Red Flags & Resource Needs", "ESI v5")
                complaint = st.text_area("Presenting Complaint / Clinical Context", value=def_complaint, height=70)

                render_html('<div class="sa-subhead">Decision Point A/B Checklist</div>')
                req_life = st.checkbox("Immediate life-saving intervention / Agonal / Pulseless (ESI 1)", value=bool(def_req_life))
                high_risk = st.checkbox("High risk presentation / Severe acute distress (ESI 2)", value=bool(def_high_risk))
                ams = st.checkbox("Altered mental status / Acute disorientation / GCS < 14 (ESI 2)", value=bool(def_ams))
                chest_pain = st.checkbox("Acute ischemic chest pain / STEMI suspect (ESI 2)", value=bool(def_chest_pain))
                stroke_sym = st.checkbox("Acute stroke symptoms / FAST+ (ESI 2)", value=bool(def_stroke))

                resources = st.selectbox(
                    "Decision Point C: Expected Resources",
                    [2, 1, 0, -1],
                    format_func=lambda x: "2+ Resources: Labs, CT/X-Ray, IV meds (ESI 3)" if x==2 else ("1 Resource: X-Ray or Simple Suturing (ESI 4)" if x==1 else ("0 Resources: Exam/Prescription Only (ESI 5)" if x==0 else "Insufficient Data (Arrival intake uncollected)")),
                    index=0 if def_resources==2 else (1 if def_resources==1 else 2)
                )

        with st.container(key="card_frailty"):
            render_html(
                '<div class="sa-card-plain-head"><div class="sa-card-title" style="font-size:16px">'
                'Clinical Frailty Scale Assessment</div></div>'
                '<div class="sa-muted" style="margin-bottom:6px">Adjust frailty baseline to calibrate age-adjusted XGBoost decision tree risk multipliers.</div>'
            )
            cfs = st.slider("Clinical Frailty Scale (CFS: 1=Robust, 9=Terminally Ill)", min_value=1, max_value=9, value=int(def_cfs))
            run_col1, run_col2 = st.columns([3, 2], vertical_alignment="center")
            with run_col1:
                render_html(
                    f'<div class="sa-muted" style="display:flex;align-items:center;gap:6px">'
                    f'{icon("verified_user", "color:#346645;font-size:16px")}'
                    f'Dual-engine inference pipeline: Demographic XGBoost Agent + Deterministic ESI v5 Safety Floor</div>'
                )
            with run_col2:
                submit_intake = st.form_submit_button("Run Analysis", type="primary",
                                                      icon=":material/bolt:", use_container_width=True)

    if submit_intake:
        patient_data = {
            'patient_id': p_id,
            'age': float(age),
            'age_cohort': cohort,
            'gender': gender,
            'cfs_frailty_score': int(cfs),
            'has_prior_history': int(has_history),
            'comorbidity_count': int(comorbidity),
            'heart_rate': float(hr),
            'resp_rate': float(rr),
            'spo2': float(spo2),
            'sbp': float(sbp),
            'temp_c': float(temp_c),
            'chief_complaint': complaint,
            'requires_lifesaving_intervention': 1 if req_life else 0,
            'high_risk_situation': 1 if high_risk else 0,
            'altered_mental_status': 1 if ams else 0,
            'acute_chest_pain_ischemic': 1 if chest_pain else 0,
            'acute_stroke_symptoms': 1 if stroke_sym else 0,
            'resources_used': None if resources == -1 else resources
        }

        with st.spinner("Executing Demographic XGBoost Agent, TreeExplainer, and ESI v5 Safety Net..."):
            analysis = st.session_state.orchestrator.analyze_patient(
                patient_data=patient_data,
                wait_time_mins=float(wait_time)
            )
            st.session_state.last_analysis = analysis

    # Render Analysis Results
    if st.session_state.last_analysis is not None:
        res = st.session_state.last_analysis
        st.write("")

        final_esi = res['final_esi']
        esi_style = ESI_STYLES.get(int(final_esi), ESI_STYLES[3])
        ag_state = res['agreement_state']
        if ag_state == "AGREEMENT":
            consensus_html = (f'<div class="sa-consensus" style="background:#dcefe0;color:#1e5031">'
                              f'{icon("check_circle")}ML &amp; Rules Consensus</div>')
            status_chip = '<span class="sa-chip sa-chip-green">Status: ML &amp; Rules Consensus</span>'
        elif ag_state == "RULE_SAFETY_ESCALATION":
            consensus_html = (f'<div class="sa-consensus" style="background:#ffdcbd;color:#693c00">'
                              f'{icon("shield")}Rule Floor Escalated</div>')
            status_chip = '<span class="sa-chip sa-chip-amber">Status: Rule Floor Escalated</span>'
        else:
            consensus_html = (f'<div class="sa-consensus" style="background:#e6edf0;color:#23404d">'
                              f'{icon("trending_up")}ML Risk Escalated</div>')
            status_chip = '<span class="sa-chip">Status: ML Risk Escalated</span>'

        section_heading("Stage 1 Output", "Validated Inference Engine",
                        "Acuity Determination & Explainability Gate", status_chip)

        # Top Metric Banner
        lock_html = (f'<span class="sa-tile-foot">{icon("lock", "color:#9f3c16")}Deterministically Safety Floor Locked</span>'
                     if res.get('is_hard_locked') else '<span class="sa-tile-foot">&nbsp;</span>')
        risk_color = "#9f3c16" if res['p_risk'] >= 0.504 else "#346645"
        risk_icon = "arrow_upward" if res['p_risk'] >= 0.504 else "arrow_downward"
        conf_val = res['confidence_score']
        if conf_val < 20.0:
            conf_foot = f'<span class="sa-tile-foot" style="color:#ba1a1a">{icon("warning")}Mandatory Review (&lt;20%)</span>'
        else:
            conf_foot = f'<span class="sa-tile-foot" style="color:#346645">{icon("check")}High Consensus</span>'

        render_html(
            '<div class="sa-metrics sa-metrics-5">'
            f'<div class="sa-tile"><span class="sa-tile-label">Final Acuity Determination</span>'
            f'<span class="sa-esi-solid" style="background:{esi_style["solid"]}">{icon("priority_high")}ESI LEVEL {final_esi}</span>'
            f'{lock_html}</div>'
            f'<div class="sa-tile"><span class="sa-tile-label">Raw Risk Probability (P_risk)</span>'
            f'<span class="sa-tile-value">{res["p_risk"]*100:.1f}%</span>'
            f'<span class="sa-tile-foot" style="color:{risk_color}">{icon(risk_icon)}Threshold: 50.4%</span></div>'
            f'<div class="sa-tile"><span class="sa-tile-label">Tree-Level Confidence</span>'
            f'<span class="sa-tile-value">{conf_val:.1f}%</span>{conf_foot}</div>'
            f'<div class="sa-tile"><span class="sa-tile-label">Stage 2 Priority Score</span>'
            f'<span class="sa-tile-value" style="color:#9f3c16">{res["priority_score"]:.1f} <small>pts</small></span>'
            f'<span class="sa-tile-foot">{icon("timer", "color:#346645")}Wait: {esc(res["wait_time_mins"])} min</span></div>'
            f'<div class="sa-tile"><span class="sa-tile-label">Consensus Gate</span>{consensus_html}'
            f'<span class="sa-tile-foot">ML ESI {res["ml_esi_recommendation"]} · Floor ESI {res["rule_acuity_floor"]}</span></div>'
            '</div>'
        )

        # Mandatory Review Alert
        if res['requires_human_review']:
            st.error("🚨 **MANDATORY HUMAN OVERRIDE CHECKPOINT**: Epistemic tree variance indicates high model uncertainty (<20% confidence) or unverified history conflict. Attendee clinician sign-off required.")

        # SHAP & Safety Rules Row
        exp_col1, exp_col2 = st.columns([3, 2])

        with exp_col1:
            with st.container(key="card_shap"):
                card_head("bar_chart", "SHAP Feature Attribution (TreeExplainer)", "Log-Odds Impact")
                top_shaps = res['shap_details'].get('top_features', [])
                if top_shaps:
                    shap_df = pd.DataFrame(top_shaps)
                    shap_df['feature_clean'] = shap_df['feature'].str.replace('_', ' ').str.title()

                    chart = alt.Chart(shap_df).mark_bar(cornerRadius=3).encode(
                        x=alt.X('shap_value:Q', title='SHAP Impact on Clinical Risk (Log-Odds)'),
                        y=alt.Y('feature_clean:N', sort='-x', title='Clinical Feature'),
                        color=alt.Color('is_risk_increasing:N', scale=alt.Scale(domain=[True, False], range=['#ba1a1a', '#346645']), legend=alt.Legend(title="Impact", labelExpr="datum.value ? 'Elevates Risk' : 'Protective / Normal'", orient="top")),
                        tooltip=['feature_clean', 'value', 'shap_value', 'cohort_band']
                    ).properties(height=260).configure(
                        background='transparent', font='Inter'
                    ).configure_axis(
                        labelColor='#57423b', titleColor='#57423b', gridColor='#f1e6dc', domainColor='#dec0b7', tickColor='#dec0b7'
                    ).configure_legend(labelColor='#57423b', titleColor='#57423b').configure_view(strokeWidth=0)
                    st.altair_chart(chart, use_container_width=True)
                render_html(
                    f'<div class="sa-footnote"><span>Model Agent: <b>{esc(str(res["ml_details"].get("cohort", "adult")).capitalize())} XGBoost</b></span>'
                    f'<span>SHAP Explainer: <b>TreeExplainer</b></span></div>'
                )

        with exp_col2:
            with st.container(key="card_rules"):
                triggered = res['rule_details'].get('triggered_rules', [])
                card_head("rule", "Deterministic ESI v5 Rules Evaluation",
                          f"{len(triggered)} Rule{'s' if len(triggered) != 1 else ''} Triggered")
                if triggered:
                    render_html("".join(
                        f'<div class="sa-rule">{icon("location_on")}<span>{esc(rule_text)}</span></div>'
                        for rule_text in triggered
                    ))
                else:
                    st.caption("No acute ABCDE red flags triggered. Acuity governed by standard resource pathways.")

                render_html(
                    f'<div class="sa-footnote"><span>ML Recommendation: <b style="color:#9f3c16">ESI {res["ml_esi_recommendation"]}</b>'
                    f' · Safety Floor: <b style="color:#9f3c16">ESI {res["rule_acuity_floor"]}</b></span>'
                    f'<span>Rule Engine Equation: Final_ESI = min(ML_ESI, ABCDE_Floor)</span></div>'
                )

        # Clinical Narrative Trace
        with st.container(key="card_narrative"):
            narrative = res.get('clinical_narrative', {})
            audit_info = res.get('grounding_audit', {})
            audit_chip = ('<span class="sa-chip sa-chip-green">● Audit Validated</span>' if audit_info.get('is_grounded')
                          else '<span class="sa-chip sa-chip-red">● Grounding Violations</span>')

            # Name the engine that wrote this narrative. A silent drop to the
            # deterministic fallback used to be invisible on screen.
            source = res.get('narrative_source', {})
            if source.get('provider') == 'ollama':
                engine_chip = (f'<span class="sa-chip sa-chip-green">◆ {esc(source.get("model", "Gemma 4"))} (local)</span>')
            else:
                engine_chip = '<span class="sa-chip sa-chip-amber">◆ Deterministic engine</span>'

            render_html(
                f'<div class="sa-card-head"><div class="sa-card-title">{icon("description")}'
                f'Grounded Clinical Reasoning Trace</div>{engine_chip}{audit_chip}</div>'
                f'<div class="sa-narrative"><em>{esc(narrative.get("clinical_rationale", "Clinical rationale synthesized."))}</em></div>'
            )

            if source.get('fallback_reason'):
                render_html(
                    f'<div class="sa-muted">Local language model unavailable '
                    f'({esc(source["fallback_reason"])}) - narrative written by the deterministic engine.</div>'
                )

            if audit_info.get('is_grounded'):
                render_html(
                    f'<div class="sa-callout sa-callout-ok">{icon("verified")}<span><b>Grounding Validator Passed</b>: '
                    f'All {len(audit_info.get("verified_facts", []))} clinical claims mathematically verified against SHAP attributions '
                    f'and raw physiological vitals.</span></div>'
                )
            else:
                render_html(
                    f'<div class="sa-callout sa-callout-bad">{icon("error")}<span><b>Grounding Violations Detected</b>: '
                    f'{esc(", ".join(audit_info.get("violations", [])))}</span></div>'
                )

            # Add to Queue Button
            add_col1, add_col2 = st.columns([3, 2], vertical_alignment="center")
            with add_col1:
                render_html('<div class="sa-muted">Next stage: the patient is added to the Queue Scheduler '
                            'and ranked by their priority score.</div>')
            with add_col2:
                if st.button("Schedule Patient into Queue", type="primary",
                             icon=":material/add_task:", use_container_width=True):
                    st.session_state.scheduler.add_patient(
                        patient_id=res['patient_id'],
                        esi_final=res['final_esi'],
                        p_risk=res['p_risk'],
                        wait_time_mins=res['wait_time_mins'],
                        age=int(res['raw_patient_data'].get('age', 35)),
                        age_cohort=res['ml_details'].get('cohort', 'adult'),
                        gender=res['raw_patient_data'].get('gender', 'Unknown'),
                        chief_complaint=res['raw_patient_data'].get('chief_complaint', 'General ED presentation'),
                        has_prior_history=int(res['raw_patient_data'].get('has_prior_history', 0)),
                        confidence_score=res['confidence_score'],
                        requires_human_review=res['requires_human_review'],
                        vital_signs={
                            'heart_rate': res['raw_patient_data'].get('heart_rate'),
                            'resp_rate': res['raw_patient_data'].get('resp_rate'),
                            'spo2': res['raw_patient_data'].get('spo2'),
                            'sbp': res['raw_patient_data'].get('sbp'),
                            'temp_c': res['raw_patient_data'].get('temp_c'),
                            'cfs_frailty_score': res['raw_patient_data'].get('cfs_frailty_score')
                        }
                    )
                    # Fresh ID for the next intake so a new patient doesn't overwrite this queue entry
                    st.session_state.intake_patient_id = f"PID_{np.random.randint(10000, 99999)}"
                    st.session_state.last_added_patient_id = res['patient_id']
                    st.session_state.schedule_message = f"Patient {res['patient_id']} added to Waiting Room Queue! Switch to Stage 2 tab to view real-time rank."
                    st.rerun()
            if st.session_state.get('schedule_message'):
                st.success(st.session_state.pop('schedule_message'))


# =========================================================================
# TAB 2: QUEUE SCHEDULER
# =========================================================================
with tab2:
    section_heading("Priority Engine Active", "Continuous Priority Scoring",
                    "Queue Scheduler")

    queue_list = st.session_state.scheduler.get_ranked_queue()

    # Summary Metrics Row
    high_acuity_count = sum(1 for p in queue_list if p['esi_final'] <= 2)
    pending_review_count = sum(1 for p in queue_list if p['requires_human_review'])
    render_html(
        '<div class="sa-metrics sa-metrics-3">'
        f'<div class="sa-tile sa-tile-accent" style="border-left-color:#9f3c16">'
        f'<span class="sa-tile-label">Total Patients Waiting {icon("group", "color:#9f3c16")}</span>'
        f'<span class="sa-tile-value">{len(queue_list)}</span><span class="sa-tile-foot">Active waiting room queue</span></div>'
        f'<div class="sa-tile sa-tile-accent" style="border-left-color:#ba1a1a">'
        f'<span class="sa-tile-label">Critical / Emergent (ESI 1-2) {icon("emergency", "color:#ba1a1a")}</span>'
        f'<span class="sa-tile-value" style="color:#ba1a1a">{high_acuity_count}</span><span class="sa-tile-foot">Highest-acuity patients in queue</span></div>'
        f'<div class="sa-tile sa-tile-accent" style="border-left-color:#346645">'
        f'<span class="sa-tile-label">Pending Nurse Review {icon("verified_user", "color:#346645")}</span>'
        f'<span class="sa-tile-value" style="color:#346645">{pending_review_count}</span><span class="sa-tile-foot">Flagged for clinician sign-off</span></div>'
        '</div>'
    )

    # Queue Management Controls
    with st.container(key="card_controls"):
        # Equal spacer columns either side keep the control group centred in the card
        _, ctrl_col1, ctrl_col2, ctrl_col3, _ = st.columns([0.7, 1.35, 0.8, 1.75, 0.7], vertical_alignment="center")
        with ctrl_col1:
            if st.button("Populate Demo Queue (12 Patients)", icon=":material/sync:", use_container_width=True):
                st.session_state.last_added_patient_id = None
                st.session_state.scheduler.clear()
                df_demo = generate_synthetic_ed_data(n_samples=12, random_seed=999)
                for _, row in df_demo.iterrows():
                    p = row.to_dict()
                    res = st.session_state.orchestrator.analyze_patient(
                        patient_data=p,
                        wait_time_mins=float(p.get('current_wait_time_mins', 5.0)),
                        use_llm=False
                    )
                    st.session_state.scheduler.add_patient(
                        patient_id=res['patient_id'],
                        esi_final=res['final_esi'],
                        p_risk=res['p_risk'],
                        wait_time_mins=res['wait_time_mins'],
                        age=int(p.get('age', 35)),
                        age_cohort=res['ml_details'].get('cohort', 'adult'),
                        gender=p.get('gender', 'Unknown'),
                        chief_complaint="ED Clinical Presentation",
                        has_prior_history=int(p.get('has_prior_history', 0)),
                        confidence_score=res['confidence_score'],
                        requires_human_review=res['requires_human_review'],
                        vital_signs={
                            'heart_rate': p.get('heart_rate'),
                            'resp_rate': p.get('resp_rate'),
                            'spo2': p.get('spo2'),
                            'sbp': p.get('sbp'),
                            'temp_c': p.get('temp_c'),
                            'cfs_frailty_score': p.get('cfs_frailty_score')
                        }
                    )
                st.rerun()
        with ctrl_col2:
            if st.button("Clear Queue", icon=":material/clear_all:", use_container_width=True):
                st.session_state.last_added_patient_id = None
                st.session_state.scheduler.clear()
                st.rerun()
        with ctrl_col3:
            with st.form("advance_time_form", clear_on_submit=True, border=False):
                time_col1, time_col2 = st.columns([1.3, 1], vertical_alignment="center")
                with time_col1:
                    advance_mins_text = st.text_input(
                        "Advance wait time (mins)",
                        placeholder="Enter minutes, e.g. 10",
                        label_visibility="collapsed"
                    )
                with time_col2:
                    advance_submitted = st.form_submit_button("Add Time", icon=":material/timer:",
                                                              type="primary", use_container_width=True)
            if advance_submitted:
                try:
                    advance_mins = float(advance_mins_text.strip())
                except ValueError:
                    advance_mins = None
                if advance_mins is None or advance_mins <= 0:
                    st.error("Enter a positive number of minutes.")
                else:
                    st.session_state.scheduler.advance_time(advance_mins)
                    st.rerun()

    st.write("")
    with st.container(key="card_queue"):
        render_html(
            '<div class="sa-card-plain-head"><div class="sa-card-title" style="font-size:22px">'
            '<span style="width:10px;height:10px;border-radius:50%;background:#9f3c16;display:inline-block"></span>'
            'Real-Time Waiting Room Queue</div>'
            '<span class="sa-chip sa-chip-terra">Ordering: Highest Priority Score First</span></div>'
        )

        if queue_list:
            # Highlight the most recently scheduled patient (until the next patient is added)
            new_pid = st.session_state.get('last_added_patient_id')
            new_entry = next((p for p in queue_list if p['patient_id'] == new_pid), None)
            if new_entry is not None:
                render_html(
                    f'<div class="sa-callout sa-callout-ok">{icon("person_add")}<span><b>Newly added:</b> Patient '
                    f'{esc(new_entry["patient_id"])} is currently ranked <b>#{new_entry["queue_rank"]}</b> of {len(queue_list)} '
                    f'(ESI {new_entry["esi_final"]}, priority score {new_entry["priority_score"]:.1f}).</span></div>'
                )

            # Display Table
            row_html = []
            for p in queue_list:
                is_root = p['queue_rank'] == 1
                is_new = new_entry is not None and p['patient_id'] == new_pid
                rank_html = (f'<span class="sa-rank sa-rank-root">#{p["queue_rank"]}</span>' if is_root
                             else f'<span class="sa-rank">#{p["queue_rank"]}</span>')
                pid_html = (f'<span class="sa-strong-terra">{esc(p["patient_id"])}</span>'
                            if is_root else esc(p['patient_id']))
                if is_new:
                    pid_html += '<span class="sa-new-tag">NEW</span>'
                row_classes = " ".join(c for c, on in (("sa-root", is_root), ("sa-new", is_new)) if on)
                review_html = ('<span class="sa-chip sa-chip-red">⚠ Review Required</span>' if p['requires_human_review']
                               else '<span class="sa-chip sa-chip-green">✓ Verified</span>')
                override_html = (f'<span class="sa-chip sa-chip-amber" style="text-transform:none">✏ Overridden ({esc(p["override_reason"])})</span>'
                                 if p['is_overridden'] else '<span class="sa-chip" style="text-transform:none">Auto-Assessed</span>')
                score_style = ' style="color:#9f3c16;font-weight:700;font-size:15px"' if is_root else ''
                row_html.append(
                    f'<tr class="{row_classes}">'
                    f'<td>{rank_html}</td>'
                    f'<td>{pid_html}</td>'
                    f'<td>{esi_soft_badge(p["esi_final"])}</td>'
                    f'<td style="color:#9f3c16;font-weight:600">{p["p_risk"]*100:.1f}%</td>'
                    f'<td>{p["wait_time_mins"]:.1f} min</td>'
                    f'<td{score_style}>{p["priority_score"]:.1f}</td>'
                    f'<td>{esc(p["age_cohort"].capitalize())} ({esc(p["age"])}y)</td>'
                    f'<td>{review_html}</td>'
                    f'<td>{override_html}</td>'
                    '</tr>'
                )
            render_html(
                '<div class="sa-table-wrap"><table class="sa-table"><thead><tr>'
                '<th>Rank</th><th>Patient ID</th><th>Acuity Level</th><th>P(Risk)</th><th>Wait Time</th>'
                '<th>Priority Score</th><th>Cohort / Age</th><th>Review Flag</th><th>Override Status</th>'
                '</tr></thead><tbody>' + "".join(row_html) + '</tbody></table></div>'
                f'<div class="sa-table-foot"><span>Patients in Queue: <b>{len(queue_list)}</b></span>'
                f'<span>Scores recalculated on every queue update</span></div>'
            )
        else:
            st.info("Waiting room queue is currently empty. Use the intake tab or 'Populate Demo Queue' button above to add patients.")

    if queue_list:
        # Actions Section: Pop next / Overrides / Deterioration
        st.write("")
        action_col1, action_col2 = st.columns(2)

        with action_col1:
            with st.container(key="card_bed"):
                card_plain_head("bed", "Bed Assignment & Examination", "Immediate dispatch of the highest-priority patient")
                top_p = queue_list[0]
                render_html(
                    '<div class="sa-candidate"><div class="sa-candidate-top"><span>Highest Priority Candidate</span><span>Rank #1</span></div>'
                    f'<div class="sa-candidate-name">Next in Line for Bed Assignment: '
                    f'<span style="color:#9f3c16">Patient {esc(top_p["patient_id"])}</span></div>'
                    f'<div class="sa-candidate-meta">{esi_soft_badge(top_p["esi_final"])}'
                    f'<span>Priority Score: <b style="color:#9f3c16">{top_p["priority_score"]:.1f}</b></span>'
                    f'<span>· {esc(top_p["age_cohort"].capitalize())} {esc(top_p["age"])}y</span></div></div>'
                )
                if st.button("Call Next Patient to Exam Bed", type="primary", icon=":material/notifications_active:",
                             use_container_width=True):
                    popped = st.session_state.scheduler.pop_next_patient()
                    st.success(f"Called {popped.patient_id} into ED examination room. Queue re-indexed.")
                    st.rerun()

        with action_col2:
            with st.container(key="card_override"):
                card_plain_head("edit_note", "Nurse Clinical Override Checkpoint",
                                "Discretionary acuity adjustment & audit trail",
                                '<span class="sa-chip sa-chip-amber" style="text-transform:none">Charge_Nurse_Lead</span>')
                patient_ids = [p['patient_id'] for p in queue_list]
                ov_col1, ov_col2 = st.columns(2)
                with ov_col1:
                    override_target_id = st.selectbox("Select Patient to Override", patient_ids)
                with ov_col2:
                    override_new_esi = st.selectbox("Assign New ESI Acuity", [1, 2, 3, 4, 5], index=1)
                override_reason = st.text_input("Override Rationale (Required for Audit Trail)", value="Clinical intuition / Observed subtle frailty")
                render_html(
                    f'<div class="sa-callout sa-callout-note">{icon("info")}<span>Override recalculates the priority score '
                    f'with the new ESI level and immediately re-sorts the queue. Every override is recorded '
                    f'in the Nurse Override Audit Log.</span></div>'
                )

                if st.button("Submit Clinician Override", type="primary", icon=":material/published_with_changes:",
                             use_container_width=True):
                    st.session_state.scheduler.record_nurse_override(
                        patient_id=override_target_id,
                        new_esi=override_new_esi,
                        reason=override_reason,
                        clinician_id="Charge_Nurse_Lead"
                    )
                    st.success(f"Override applied! Patient {override_target_id} updated to ESI {override_new_esi} and queue re-sorted.")
                    st.rerun()


# =========================================================================
# TAB 3: NURSE OVERRIDE AUDIT LOG
# =========================================================================
with tab3:
    section_heading("Audit Trail", "Immutable clinician override record", "Nurse Override Audit Log")
    with st.container(key="card_audit"):
        audit_log = st.session_state.scheduler.override_audit_log
        if audit_log:
            audit_df = pd.DataFrame(audit_log)
            header_html = "".join(f"<th>{esc(col.replace('_', ' '))}</th>" for col in audit_df.columns)
            body_rows = []
            for idx, row in audit_df.iterrows():
                cells = []
                for col in audit_df.columns:
                    val = row[col]
                    if col in ('original_esi', 'overridden_esi'):
                        cells.append(f'<td>{esi_soft_badge(val)}</td>')
                    elif col == 'resulting_score':
                        cells.append(f'<td class="sa-strong-terra">{esc(val)}</td>')
                    elif col == 'clinician_id':
                        cells.append(f'<td style="color:#9f3c16">{esc(val)}</td>')
                    else:
                        cells.append(f'<td>{esc(val)}</td>')
                body_rows.append(f'<tr><td>{idx}</td>{"".join(cells)}</tr>')
            render_html(
                '<div class="sa-table-wrap"><table class="sa-table"><thead><tr><th>#</th>' + header_html +
                '</tr></thead><tbody>' + "".join(body_rows) + '</tbody></table></div>'
            )
        else:
            st.caption("No manual clinician overrides logged in current session.")
