"""
Decision Point B: High-Risk Situation / Altered Mental Status / Severe Pain or Distress (ESI Level 2)
Evaluates whether the patient is in a high-risk scenario that cannot wait safely.
"""

from typing import Dict, Any, Tuple, List


def check_decision_point_b(patient: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if patient triggers Decision Point B criteria for ESI 2.
    
    Criteria:
    - high_risk_situation == 1
    - altered_mental_status == 1 (or AVPU in ['V', 'P'] or GCS < 14)
    - severe_pain_distress == 1 (or pain_score >= 8 with high-risk clinical context)
    - acute_chest_pain_ischemic == 1
    - acute_stroke_symptoms == 1
    - anaphylaxis_early == 1
    - acute_suicidal_ideation == 1
    
    Returns
    -------
    (is_esi_2, triggered_reasons)
    """
    triggers = []

    # 1. High risk situation generic flag
    if patient.get('high_risk_situation') in (1, True, '1', 'true', 'True'):
        triggers.append("Decision Point B: High-risk clinical situation identified")

    # 2. Altered mental status / Lethargy / Confusion
    avpu = str(patient.get('avpu', '')).upper()
    gcs = patient.get('gcs')
    if patient.get('altered_mental_status') in (1, True, '1', 'true', 'True'):
        triggers.append("Decision Point B: Altered mental status / acute confusion / disorientation")
    elif avpu in ('V', 'P'):
        triggers.append(f"Decision Point B: Altered responsiveness (AVPU={avpu})")
    elif gcs is not None and 8 < gcs < 14:
        triggers.append(f"Decision Point B: Depressed Glasgow Coma Scale (GCS={gcs})")

    # 3. Severe pain / distress
    pain_score = patient.get('pain_score')
    if patient.get('severe_pain_distress') in (1, True, '1', 'true', 'True'):
        triggers.append("Decision Point B: Severe pain or severe clinical distress")
    elif pain_score is not None and pain_score >= 8 and patient.get('severe_distress') in (1, True):
        triggers.append(f"Decision Point B: Severe excruciating pain (Score {pain_score}/10) with systemic distress")

    # 4. Acute Ischemic Chest Pain / STEMI suspect
    if patient.get('acute_chest_pain_ischemic') in (1, True, '1', 'true', 'True'):
        triggers.append("Decision Point B: Acute ischemic chest pain / suspected acute coronary syndrome")

    # 5. Acute Stroke / Neurological Deficit
    if patient.get('acute_stroke_symptoms') in (1, True, '1', 'true', 'True'):
        triggers.append("Decision Point B: Acute stroke symptoms / sudden focal neurological deficit")

    # 6. Anaphylaxis / Severe allergic response
    if patient.get('anaphylaxis_early') in (1, True, '1', 'true', 'True'):
        triggers.append("Decision Point B: Systemic allergic reaction / evolving anaphylaxis")

    # 7. Acute psychiatric / behavioral safety crisis
    if patient.get('acute_suicidal_ideation') in (1, True, '1', 'true', 'True') or patient.get('suicidal_ideation_acute') in (1, True):
        triggers.append("Decision Point B: Acute psychiatric danger / suicidal ideation with active intent")

    is_esi_2 = len(triggers) > 0
    return is_esi_2, triggers

