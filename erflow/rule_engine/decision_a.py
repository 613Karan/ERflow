"""
Decision Point A: Immediate Life-Saving Intervention Required (ESI Level 1)
Evaluates whether the patient has an airway, breathing, or circulatory emergency
demanding immediate resuscitation.
"""

from typing import Dict, Any, Tuple, List


def check_decision_point_a(patient: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if patient triggers Decision Point A criteria for ESI 1.
    
    Criteria:
    - requires_lifesaving_intervention == 1
    - intubation / acute airway compromise
    - cardiopulmonary_arrest / pulseless / apneic
    - severe respiratory distress (agonal)
    - acute profound unresponsiveness (AVPU = 'U' or GCS <= 8)
    
    Returns
    -------
    (is_esi_1, triggered_reasons)
    """
    triggers = []
    
    # 1. Direct explicit flag
    if patient.get('requires_lifesaving_intervention') in (1, True, '1', 'true', 'True'):
        triggers.append("Decision Point A: Immediate life-saving intervention requested / required")
        
    # 2. Airway / Breathing compromise
    if patient.get('intubation') in (1, True, '1', 'true', 'True') or patient.get('severe_respiratory_distress_agonal') in (1, True):
        triggers.append("Decision Point A: Severe respiratory distress / agonal respiration / airway compromise")

    # 3. Cardiac arrest / Pulselessness
    if patient.get('cardiopulmonary_arrest') in (1, True) or patient.get('unresponsive_apneic_pulseless') in (1, True):
        triggers.append("Decision Point A: Apneic / Pulseless / Cardiac arrest presentation")

    # 4. Neurological status: Unresponsive
    avpu = str(patient.get('avpu', '')).upper()
    gcs = patient.get('gcs')
    if avpu == 'U' or (gcs is not None and gcs <= 8 and gcs > 0):
        triggers.append(f"Decision Point A: Acute profound unresponsiveness (AVPU={avpu}, GCS={gcs})")

    # 5. Extreme vital derangement incompatible with stability
    hr = patient.get('heart_rate')
    sbp = patient.get('sbp')
    if hr is not None and (hr < 30 or hr > 220):
        triggers.append(f"Decision Point A: Critical extreme heart rate ({hr} bpm)")
    if sbp is not None and sbp < 50:
        triggers.append(f"Decision Point A: Profound hypotensive shock (SBP {sbp} mmHg)")

    is_esi_1 = len(triggers) > 0
    return is_esi_1, triggers

