"""
Decision Point C: Resource Utilization Estimation & Fallback Handler (ESI Levels 3, 4, 5)
Evaluates expected healthcare resource consumption (labs, imaging, IV meds, procedures).
If resource information is absent at intake, gracefully stubs as 'insufficient_data'.
"""

from typing import Dict, Any, Tuple, Optional, List


def estimate_decision_point_c(patient: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    """
    Estimate ESI level based on resource needs.
    
    ESI v5 Resource Guidelines:
    - 0 resources -> ESI 5 (Non-urgent)
    - 1 resource -> ESI 4 (Less urgent)
    - 2+ resources -> ESI 3 (Urgent)
    
    If resource fields are absent, return (3, 'insufficient_data', reasons).
    
    Returns
    -------
    (esi_level, data_status, rationale_list)
    """
    rationale = []
    
    # Check explicit resource count
    res_count = patient.get('resources_used')
    if res_count is None:
        res_count = patient.get('expected_resources')
        
    if res_count is not None:
        try:
            r = int(res_count)
            if r >= 2:
                rationale.append(f"Decision Point C: Multiple healthcare resources expected ({r} resources -> ESI 3)")
                return 3, "sufficient_data", rationale
            elif r == 1:
                rationale.append("Decision Point C: Single healthcare resource expected (1 resource -> ESI 4)")
                return 4, "sufficient_data", rationale
            else:
                rationale.append("Decision Point C: Zero healthcare resources required (0 resources -> ESI 5)")
                return 5, "sufficient_data", rationale
        except (ValueError, TypeError):
            pass

    # Check granular resource flags if present
    counted_resources = 0
    resource_items = []
    
    if patient.get('labs_ordered') in (1, True):
        counted_resources += 1
        resource_items.append("Labs (blood/urine)")
    if patient.get('imaging_ordered') in (1, True) or patient.get('xray_or_ct') in (1, True):
        counted_resources += 1
        resource_items.append("Imaging (X-ray/CT/Ultrasound)")
    if patient.get('iv_fluids') in (1, True) or patient.get('iv_meds') in (1, True):
        counted_resources += 1
        resource_items.append("IV Hydration / IV Medications")
    if patient.get('specialty_consult') in (1, True):
        counted_resources += 1
        resource_items.append("Specialty Consultation")
    if patient.get('procedure_required') in (1, True):
        counted_resources += 1
        resource_items.append("Clinical Procedure / Suturing")

    if resource_items:
        if counted_resources >= 2:
            rationale.append(f"Decision Point C: {counted_resources} resources identified ({', '.join(resource_items)}) -> ESI 3")
            return 3, "sufficient_data", rationale
        elif counted_resources == 1:
            rationale.append(f"Decision Point C: 1 resource identified ({resource_items[0]}) -> ESI 4")
            return 4, "sufficient_data", rationale
        else:
            rationale.append("Decision Point C: No resources identified -> ESI 5")
            return 5, "sufficient_data", rationale

    # If no resource data is available at arrival time:
    rationale.append("Decision Point C: Intake resource utilization uncollected -> stubs as 'insufficient_data' (defaults to standard ESI 3)")
    return 3, "insufficient_data", rationale

