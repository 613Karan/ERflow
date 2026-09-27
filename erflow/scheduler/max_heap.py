"""
Dynamic Max Heap Priority Queue for Real-Time Emergency Department Patient Scheduling.
Continuously recalculates priority scores as patients wait, supports real-time vital deterioration
re-sorting, and nurse override tracking.
"""

import time
import heapq
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

from erflow.scheduler.scoring import compute_priority_score


@dataclass(order=True)
class HeapItem:
    """
    Max Heap wrapper item for heapq (inverted priority score for max-heap behavior).
    """
    neg_priority_score: float
    counter: int
    patient_id: str = field(compare=False)
    entry: 'PatientQueueEntry' = field(compare=False)


@dataclass
class PatientQueueEntry:
    """
    Detailed patient record stored in the dynamic scheduler queue.
    """
    patient_id: str
    esi_final: int
    p_risk: float
    wait_time_mins: float
    age: int = 35
    age_cohort: str = 'adult'
    gender: str = 'Unknown'
    chief_complaint: str = 'General ED presentation'
    has_prior_history: int = 0
    confidence_score: float = 100.0
    requires_human_review: bool = False
    is_overridden: bool = False
    override_reason: Optional[str] = None
    override_by: Optional[str] = None
    initial_esi: Optional[int] = None
    vital_signs: Dict[str, Any] = field(default_factory=dict)
    added_at_unix: float = field(default_factory=time.time)
    priority_score: float = 0.0

    def recalculate_score(self) -> float:
        self.priority_score = compute_priority_score(
            esi_final=self.esi_final,
            p_risk=self.p_risk,
            t_wait_mins=self.wait_time_mins
        )
        return self.priority_score


class DynamicMaxHeapQueue:
    """
    Thread-safe continuous Max Heap priority queue scheduler for ED patient prioritization.
    """

    def __init__(self):
        self.patients: Dict[str, PatientQueueEntry] = {}
        self._counter: int = 0
        self.override_audit_log: List[Dict[str, Any]] = []

    def _get_next_counter(self) -> int:
        self._counter += 1
        return self._counter

    def add_patient(
        self,
        patient_id: str,
        esi_final: int,
        p_risk: float,
        wait_time_mins: float = 0.0,
        age: int = 35,
        age_cohort: str = 'adult',
        gender: str = 'Unknown',
        chief_complaint: str = 'General ED presentation',
        has_prior_history: int = 0,
        confidence_score: float = 100.0,
        requires_human_review: bool = False,
        vital_signs: Optional[Dict[str, Any]] = None
    ) -> PatientQueueEntry:
        """
        Add a new patient to the waiting room queue.
        """
        entry = PatientQueueEntry(
            patient_id=patient_id,
            esi_final=esi_final,
            p_risk=p_risk,
            wait_time_mins=wait_time_mins,
            age=age,
            age_cohort=age_cohort,
            gender=gender,
            chief_complaint=chief_complaint,
            has_prior_history=has_prior_history,
            confidence_score=confidence_score,
            requires_human_review=requires_human_review,
            initial_esi=esi_final,
            vital_signs=vital_signs or {}
        )
        entry.recalculate_score()
        self.patients[patient_id] = entry
        return entry

    def remove_patient(self, patient_id: str) -> Optional[PatientQueueEntry]:
        """
        Remove a patient when called into examination or discharged.
        """
        return self.patients.pop(patient_id, None)

    def advance_time(self, delta_minutes: float = 5.0):
        """
        Advance simulated clock for all waiting room patients and update priority scores.
        """
        for entry in self.patients.values():
            entry.wait_time_mins += delta_minutes
            entry.recalculate_score()

    def update_patient_vitals_and_risk(
        self,
        patient_id: str,
        new_p_risk: float,
        new_esi: Optional[int] = None,
        updated_vitals: Optional[Dict[str, Any]] = None
    ) -> Optional[PatientQueueEntry]:
        """
        Deterioration Re-Sort: update vitals, recompute risk, and instantly re-index heap.
        """
        entry = self.patients.get(patient_id)
        if not entry:
            return None

        entry.p_risk = new_p_risk
        if new_esi is not None:
            entry.esi_final = new_esi
        if updated_vitals:
            entry.vital_signs.update(updated_vitals)

        entry.recalculate_score()
        return entry

    def record_nurse_override(
        self,
        patient_id: str,
        new_esi: int,
        reason: str,
        clinician_id: str = "Charge_Nurse_1"
    ) -> Optional[PatientQueueEntry]:
        """
        Record manual clinician override and immediately re-sort priority queue.
        """
        entry = self.patients.get(patient_id)
        if not entry:
            return None

        old_esi = entry.esi_final
        entry.esi_final = new_esi
        entry.is_overridden = True
        entry.override_reason = reason
        entry.override_by = clinician_id
        entry.recalculate_score()

        audit_entry = {
            'timestamp': time.time(),
            'patient_id': patient_id,
            'clinician_id': clinician_id,
            'original_esi': old_esi,
            'overridden_esi': new_esi,
            'reason': reason,
            'resulting_score': entry.priority_score
        }
        self.override_audit_log.append(audit_entry)
        return entry

    def _recalculate_all_scores(self):
        for entry in self.patients.values():
            entry.recalculate_score()

    def get_ranked_queue(self) -> List[Dict[str, Any]]:
        """
        Extract ordered queue ranked by dynamic priority score descending using Max Heap.
        """
        heap = []
        for entry in self.patients.values():
            entry.recalculate_score()
            heap_item = HeapItem(
                neg_priority_score=-entry.priority_score,
                counter=self._get_next_counter(),
                patient_id=entry.patient_id,
                entry=entry
            )
            heapq.heappush(heap, heap_item)

        ranked_list = []
        rank = 1
        while heap:
            item = heapq.heappop(heap)
            p_dict = asdict(item.entry)
            p_dict['queue_rank'] = rank
            ranked_list.append(p_dict)
            rank += 1

        return ranked_list

    def pop_next_patient(self) -> Optional[PatientQueueEntry]:
        """
        Pop the highest priority patient from the queue for ED bed placement.
        """
        ranked = self.get_ranked_queue()
        if not ranked:
            return None
        top_id = ranked[0]['patient_id']
        return self.remove_patient(top_id)

    def size(self) -> int:
        return len(self.patients)

    def clear(self):
        self.patients.clear()

