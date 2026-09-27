"""
Ollama API Client Wrapper with Fallback Local Reasoning Engine.
Connects to a local Ollama instance (gemma4:12b) for grounded clinical narrative
synthesis, with a seamless zero-dependency deterministic fallback whenever Ollama
is offline, too slow, or returns output the grounding contract cannot accept.

The language model never decides acuity. The XGBoost models and the ESI v5 rule
engine do that; the model only explains a result that has already been computed,
and every narrative is checked by GroundingValidator before display.
"""

import os
import json
import socket
import logging
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)

# Keys the rest of the system reads off a narrative. grounding.py checks
# 'recommended_esi', 'primary_risk_factors' and 'triggered_safety_rules';
# ui/app.py renders 'clinical_rationale'. The LLM path must produce all four or
# it is rejected in favour of the deterministic engine.
REQUIRED_NARRATIVE_KEYS = (
    'recommended_esi',
    'clinical_rationale',
    'primary_risk_factors',
    'triggered_safety_rules',
)

ESI_LABELS = {
    1: "ESI 1 (Resuscitation - Immediate Life-Saving Intervention Required)",
    2: "ESI 2 (Emergent / High Risk - Escalated Priority)",
    3: "ESI 3 (Urgent - Multiple Resources / Standard Priority)",
    4: "ESI 4 (Less Urgent - Single Resource)",
    5: "ESI 5 (Non-Urgent - Zero Resources)",
}

GOVERNANCE_DISCLAIMER = (
    "DECISION SUPPORT NOTICE: ERflow provides contextualized risk estimates and "
    "safety floor bounds to assist patient prioritization. Final acuity classification "
    "and clinical disposition remain the sole responsibility of the attending "
    "healthcare professional."
)


class OllamaLLMClient:
    """
    Client for interacting with local Ollama service for grounded clinical narrative synthesis.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4:12b",
        timeout: int = 90
    ):
        self.base_url = os.environ.get("OLLAMA_HOST", base_url).rstrip('/')
        self.model = os.environ.get("OLLAMA_MODEL", model)
        # A 12B model emitting a few hundred tokens of JSON on laptop hardware
        # runs well past 15s, which silently routed every request to the fallback.
        self.timeout = int(os.environ.get("OLLAMA_TIMEOUT", timeout))
        self._host, self._port = self._parse_host_port(self.base_url)
        # 'think' is a top-level chat field. Some Ollama builds reject it for
        # models that cannot think, so a 400 downgrades us for the session.
        self._send_think = True
        self.last_error: Optional[str] = None
        self.is_connected = self._test_connection()

    @staticmethod
    def _parse_host_port(base_url: str) -> tuple:
        """Resolve the probe target from base_url so OLLAMA_HOST is honoured."""
        parsed = urlparse(base_url if '//' in base_url else f'//{base_url}')
        return (parsed.hostname or '127.0.0.1'), (parsed.port or 11434)

    def _test_connection(self) -> bool:
        try:
            with socket.create_connection((self._host, self._port), timeout=0.5):
                return True
        except OSError as exc:
            logger.debug("Ollama probe failed for %s:%s - %s", self._host, self._port, exc)
            return False

    def _build_system_prompt(
        self,
        allowed_features: List[str],
        final_esi: int,
        acuity_floor: int
    ) -> str:
        """
        Spell out the exact JSON contract. The previous prompt asked for "a
        structured JSON summary" without naming a single field, so nothing
        downstream could read the result.
        """
        return (
            "You are ERflow, a clinical decision support assistant for emergency department "
            "patient prioritization. You do NOT decide acuity: the XGBoost models and the "
            "ESI v5 rule engine have already done that. Your only job is to explain their "
            "result faithfully.\n\n"
            "GROUNDING RULES - every one is machine-checked, and a violation discards your output:\n"
            f"1. 'recommended_esi' MUST be exactly {final_esi}. It may never be a less urgent "
            f"(higher) number than the deterministic safety floor of ESI {acuity_floor}.\n"
            f"2. 'primary_risk_factors' may only name features from this list: "
            f"{', '.join(allowed_features)}. Inventing any other feature name is a hallucination. "
            "Copy each feature's 'value' and its 'direction' exactly as given in shap_result "
            "('increasing' when is_risk_increasing is true, otherwise 'decreasing').\n"
            "3. 'triggered_safety_rules' MUST be copied verbatim from rule_result.triggered_rules. "
            "Do not paraphrase, merge, or add rules.\n"
            "4. Cite no vital sign value that does not appear in the input.\n\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            "{\n"
            '  "recommended_esi": <int 1-5>,\n'
            '  "clinical_rationale": "<2-4 sentence clinical prose explaining the acuity, '
            'naming the top SHAP drivers with their values and the rules that fired>",\n'
            '  "primary_risk_factors": [{"feature": "<name>", "value": <number>, '
            '"direction": "increasing" or "decreasing", "shap_impact": <number>}],\n'
            '  "triggered_safety_rules": ["<verbatim rule string>"],\n'
            '  "alignment_status": "<one sentence on whether the ML model and the ESI v5 rules agreed>"\n'
            "}\n"
            "No markdown, no commentary, no extra keys."
        )

    def generate_clinical_trace(
        self,
        patient_data: Dict[str, Any],
        ml_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        shap_result: Dict[str, Any],
        final_esi: int,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Generate grounded clinical reasoning trace using Ollama or fallback reasoner.

        use_llm=False skips the model entirely and uses the deterministic engine,
        for bulk callers that never display the prose.

        Always returns a dict carrying 'provider' ('ollama' or
        'deterministic_clinical_engine'), 'model', 'structured_narrative' and, when
        the LLM path was attempted and failed, 'fallback_reason'.
        """
        if not use_llm:
            fallback = self._generate_deterministic_clinical_trace(
                patient_data, ml_result, rule_result, shap_result, final_esi
            )
            fallback['fallback_reason'] = 'llm bypassed for bulk assessment'
            return fallback

        # Re-probe on demand: the dashboard is commonly started before Ollama.
        if not self.is_connected:
            self.is_connected = self._test_connection()

        if self.is_connected:
            narrative, error = self._request_llm_narrative(
                patient_data, ml_result, rule_result, shap_result, final_esi
            )
            if narrative is not None:
                self.last_error = None
                return {
                    'provider': 'ollama',
                    'model': self.model,
                    'structured_narrative': narrative
                }
            reason = error
            logger.warning("Ollama narrative unavailable (%s); using deterministic engine.", reason)
        else:
            reason = f"Ollama not reachable at {self._host}:{self._port}"
            logger.info("%s; using deterministic engine.", reason)

        self.last_error = reason
        fallback = self._generate_deterministic_clinical_trace(
            patient_data, ml_result, rule_result, shap_result, final_esi
        )
        fallback['fallback_reason'] = reason
        return fallback

    def _request_llm_narrative(
        self,
        patient_data: Dict[str, Any],
        ml_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        shap_result: Dict[str, Any],
        final_esi: int
    ) -> tuple:
        """
        Query Ollama. Returns (narrative, None) on success or (None, reason).
        """
        allowed_features = [
            f['feature'] for f in shap_result.get('all_features_shap', [])
        ] or list(ml_result.get('features_used', []))
        acuity_floor = rule_result.get('acuity_floor', final_esi)

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt(
                        allowed_features, final_esi, acuity_floor
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "patient": patient_data,
                        "ml_result": ml_result,
                        "rule_result": rule_result,
                        "shap_result": shap_result,
                        "final_esi": final_esi
                    })
                }
            ]
        }
        # Disabling the thinking phase is what keeps a 12B model inside the
        # request timeout; as an 'options' entry it was silently ignored.
        if self._send_think:
            payload["think"] = False

        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.exceptions.Timeout:
            return None, f"timeout after {self.timeout}s"
        except requests.exceptions.RequestException as exc:
            return None, f"transport error: {exc}"

        if response.status_code == 400 and self._send_think:
            # This Ollama build will not accept 'think' for this model.
            logger.debug("Ollama rejected top-level 'think'; retrying without it.")
            self._send_think = False
            return self._request_llm_narrative(
                patient_data, ml_result, rule_result, shap_result, final_esi
            )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        message = response.json().get("message", {})
        content = (message.get("content") or "").strip()
        if not content:
            # Output landed in the thinking channel instead of the answer.
            return None, "empty content (model returned only a thinking block)"

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            return None, f"unparseable JSON: {exc}"

        if not isinstance(parsed, dict):
            return None, f"expected a JSON object, got {type(parsed).__name__}"

        return self._coerce_llm_narrative(parsed, final_esi, ml_result)

    def _coerce_llm_narrative(
        self,
        parsed: Dict[str, Any],
        final_esi: int,
        ml_result: Dict[str, Any]
    ) -> tuple:
        """
        Normalise model output onto the narrative contract. Missing required keys
        mean the fallback runs instead. The claimed ESI is left exactly as the
        model stated it, so GroundingValidator can still catch an unsafe claim.
        """
        # Accept the legacy alias, then normalise onto the canonical key.
        if 'recommended_esi' not in parsed and 'final_esi_recommended' in parsed:
            parsed['recommended_esi'] = parsed.pop('final_esi_recommended')

        missing = [k for k in REQUIRED_NARRATIVE_KEYS if k not in parsed]
        if missing:
            return None, f"missing required keys: {', '.join(missing)}"

        try:
            parsed['recommended_esi'] = int(parsed['recommended_esi'])
        except (TypeError, ValueError):
            return None, f"recommended_esi is not an integer: {parsed['recommended_esi']!r}"

        if not isinstance(parsed.get('primary_risk_factors'), list):
            return None, "primary_risk_factors is not a list"
        if not isinstance(parsed.get('triggered_safety_rules'), list):
            return None, "triggered_safety_rules is not a list"

        # Non-clinical presentation fields are ours, not the model's.
        parsed['final_esi_recommended'] = parsed['recommended_esi']
        parsed['esi_label'] = ESI_LABELS.get(final_esi, f"ESI {final_esi}")
        parsed['risk_probability_pct'] = round(ml_result.get('p_risk', 0.0) * 100.0, 1)
        parsed['confidence_score_pct'] = ml_result.get('confidence_score', 100.0)
        parsed['requires_human_review'] = ml_result.get('requires_human_review', False)
        parsed['governance_disclaimer'] = GOVERNANCE_DISCLAIMER
        return parsed, None

    def _generate_deterministic_clinical_trace(
        self,
        patient_data: Dict[str, Any],
        ml_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        shap_result: Dict[str, Any],
        final_esi: int
    ) -> Dict[str, Any]:
        """
        Deterministic, rigorously grounded template synthesis.
        """
        age = patient_data.get('age', 35)
        cohort = ml_result.get('cohort', 'adult')
        p_risk = ml_result.get('p_risk', 0.0)
        ml_esi = ml_result.get('ml_esi_recommendation', 3)
        rule_floor = rule_result.get('acuity_floor', 5)
        triggered_rules = rule_result.get('triggered_rules', [])
        conf_score = ml_result.get('confidence_score', 100.0)
        req_review = ml_result.get('requires_human_review', False)

        top_features = shap_result.get('top_features', [])

        # Check alignment between ML and Rules
        if ml_esi == rule_floor:
            alignment_status = f"Consensus Alignment: Both ML model and ESI v5 deterministic rules agree on ESI {final_esi}."
        elif rule_floor < ml_esi:
            alignment_status = (
                f"Safety Floor Escalation: Deterministic clinical rule floor (ESI {rule_floor}) automatically "
                f"overrode ML prediction (ESI {ml_esi}) to prevent under-prioritization."
            )
        else:
            alignment_status = (
                f"ML Risk Escalation: Demographic XGBoost agent identified elevated critical risk "
                f"(P_risk={p_risk*100:.1f}%), escalating acuity to ESI {ml_esi}."
            )

        # Build feature rationale sentences
        feature_bulletins = []
        for f in top_features[:3]:
            feat = f['feature']
            val = f['value']
            shap_val = f['shap_value']
            ref = f.get('clinical_reference')
            dir_text = "elevating clinical risk" if shap_val > 0 else "reducing baseline risk"
            ref_text = f" (normal reference: {ref[0]}-{ref[1]})" if ref and isinstance(ref, list) else ""
            feature_bulletins.append(f"{feat.replace('_', ' ').title()} recorded at {val}{ref_text}, {dir_text} (SHAP: {shap_val:+.3f}).")

        trace_summary = {
            # Canonical key read by GroundingValidator; the legacy name is kept
            # as an alias so older artifacts keep resolving.
            'recommended_esi': final_esi,
            'final_esi_recommended': final_esi,
            'esi_label': ESI_LABELS.get(final_esi, f"ESI {final_esi}"),
            'alignment_status': alignment_status,
            'risk_probability_pct': round(p_risk * 100.0, 1),
            'confidence_score_pct': conf_score,
            'requires_human_review': req_review,
            'primary_risk_factors': [
                {
                    'feature': f['feature'],
                    'value': f['value'],
                    'direction': 'increasing' if f['is_risk_increasing'] else 'decreasing',
                    'shap_impact': f['shap_value']
                }
                for f in top_features
            ],
            'triggered_safety_rules': triggered_rules,
            'clinical_rationale': (
                f"Patient ({age}y, {cohort}) evaluated under {cohort.capitalize()} XGBoost Agent and ESI v5 safety guidelines. "
                f"{alignment_status} " + " ".join(feature_bulletins)
            ),
            'governance_disclaimer': GOVERNANCE_DISCLAIMER
        }

        return {
            'provider': 'deterministic_clinical_engine',
            'model': 'esi_v5_grounded_synthesizer',
            'structured_narrative': trace_summary
        }
