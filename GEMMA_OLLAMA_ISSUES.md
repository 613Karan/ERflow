# Known Issues — Gemma 4 / Ollama Integration

**Status:** open, not fixed. Documented for later action.
**Component:** `erflow/agent/llm_client.py` (with knock-on effects in `erflow/explain/grounding.py` and `erflow/ui/app.py`)
**Audited:** 26 Sep 2026

---

## Context

The local language model writes the *Grounded Clinical Reasoning Trace* shown on Stage 1. It does not decide acuity — the XGBoost models and the ESI v5 rule engine do that, and the LLM only explains the result. If the LLM path fails, a deterministic fallback engine produces the narrative instead.

**The Gemma path has never actually run on this machine.** Ollama is installed (v0.24.0) but has never been started, and no model has been pulled (`~/.ollama` does not exist). Every narrative produced so far has come from the deterministic fallback. The issues below are therefore latent: they will surface the first time Ollama is actually running.

The transport layer itself is fine — correct endpoint (`POST /api/chat`), correct message shape, sensible fallback. The problems are in the prompt contract, the key names, and the silent failure handling.

---

## Issue 1 — The prompt never specifies the required JSON fields

**Severity: high.** This is the one that undermines the safety demo.

`llm_client.py:55-60` ends the system prompt with *"Return a structured JSON summary"* but never names a single field. Meanwhile the rest of the system reads exact keys:

| Key | Read by |
|---|---|
| `clinical_rationale` | `ui/app.py:705` |
| `recommended_esi` | `explain/grounding.py:44` |
| `primary_risk_factors` | `explain/grounding.py:63` |
| `triggered_safety_rules` | `explain/grounding.py:97` |

Gemma will return plausible-looking JSON using whatever key names it invents. When they don't match:

1. The reasoning box shows the placeholder *"Clinical rationale synthesized."* (the `.get()` default at `ui/app.py:705`).
2. **The grounding validator passes while checking nothing.** Each validation loop iterates an empty list, no violations accumulate, `is_valid` returns `True`, and the green banner reads *"Grounding Validator Passed: All 0 clinical claims mathematically verified"* (`ui/app.py:710-711`).

The `0` in that message is the diagnostic tell — it means the anti-hallucination check inspected nothing at all.

**Suggested fix:** state the exact output schema in the system prompt (field names, types, and that `primary_risk_factors` entries need `feature` / `value` / `direction` / `shap_impact`). Mirror the structure the fallback engine already produces at `llm_client.py:161-187`.

---

## Issue 2 — Key-name mismatch that is already live today

**Severity: high.** Affects the current build, not just the Gemma path.

The validator looks for `recommended_esi` (`grounding.py:44`), but the fallback engine emits `final_esi_recommended` (`llm_client.py:162`) — same value, different name.

Consequence: **the acuity-floor integrity check never executes.** It is effectively dead code on both the LLM and fallback paths. This is the check that rejects a narrative claiming an ESI less urgent than the deterministic safety floor.

The other two checks (SHAP direction, rule citation) *do* run on the fallback path, because `primary_risk_factors` (line 168) and `triggered_safety_rules` (line 177) match correctly.

**Why the tests miss it:** `tests/test_grounding.py` constructs its own input dictionaries using the correct `recommended_esi` key (lines 48, 71, 90, 107). It validates the validator in isolation; nothing asserts that the narrative producer and the validator agree on key names.

**Suggested fix:** settle on one name and use it in both places. Add a test that feeds real `generate_clinical_trace()` output into the validator, so the two halves can never drift apart again.

---

## Issue 3 — `think: false` is in the wrong place

**Severity: low.** Cosmetic in effect, but the documentation built on it was wrong.

```python
# llm_client.py:66-69
"options": {
    "temperature": 0.1,
    "think": False
}
```

In Ollama's chat API, `think` is a **top-level** request field. `options` carries runtime model parameters (temperature, top_p, num_ctx…), and unrecognised keys there are ignored. The setting does nothing as written.

In practice this likely has no impact — Gemma is not a reasoning model, so there is no thinking phase to disable. The real defect was the claim attached to it: the module docstring described it as a VRAM optimisation, which was never accurate.

**Caution if fixing:** moving `think` to the top level is not automatically safe. Some Ollama versions return a 400 error when `think` is sent to a model that does not support thinking. Verify against the installed version before changing it, or simply delete the key.

---

## Issue 4 — The connection check ignores `OLLAMA_HOST`

**Severity: medium.**

The constructor honours the environment variable:

```python
# llm_client.py:24
self.base_url = os.environ.get("OLLAMA_HOST", base_url).rstrip('/')
```

but the readiness probe hardcodes the address:

```python
# llm_client.py:35
res = s.connect_ex(('127.0.0.1', 11434))
```

Point `OLLAMA_HOST` at any other host or port and the probe fails, `is_connected` stays `False`, and the app uses the fallback permanently — even though the configured server is running and reachable.

Two secondary problems in the same method:

- **50 ms timeout** (`llm_client.py:33`) is tight enough to produce false negatives on a loaded machine.
- **Runs once, at construction time** (`llm_client.py:27`). Start Ollama after the dashboard and it will not be detected until the dashboard restarts.

**Suggested fix:** parse host and port from `self.base_url`, raise the timeout to ~500 ms, and consider re-checking on demand rather than only at startup.

---

## Issue 5 — Failures are invisible

**Severity: medium.** This is what makes all of the above hard to notice.

```python
# llm_client.py:95-96
except Exception:
    pass
```

Every failure mode — server down, unknown model tag, timeout, malformed JSON — is swallowed identically, and the fallback runs. Nothing is logged.

The return value does carry `provider` (`"ollama"` or `"deterministic_clinical_engine"`) and `model`, but **the dashboard never displays either**. So there is no way to tell from the screen which engine produced the text.

Also worth reviewing: the request timeout is **15 seconds** (`llm_client.py:22`). A 12B model generating a few hundred tokens of JSON on laptop hardware can exceed that, which would silently route to the fallback on every request.

**Suggested fix:** log the exception (even at debug level), and surface the provider in the Stage 1 UI — a small badge reading "Gemma 4 (local)" vs "Deterministic engine" would make the whole class of problem self-evident during a demo.

---

## Issue 6 — Verify the model tag exists

**Severity: unknown, needs checking.**

The default is `gemma4:12b` (`llm_client.py:21`). Because nothing has ever been pulled on this machine, that tag has never been resolved against Ollama's library. Confirm before relying on it:

```bash
ollama pull gemma4:12b
```

If the tag does not exist, the pull will say so, and the correct one can be supplied via the `OLLAMA_MODEL` environment variable without any code change.

---

## Suggested order of work

1. **Issue 2** — one-line key fix, repairs a safety check that is broken right now.
2. **Issue 1** — write the schema into the prompt; this is what makes Gemma's output actually usable.
3. **Issue 5** — surface the provider in the UI, so the remaining issues become visible instead of silent.
4. **Issue 4** — honour `OLLAMA_HOST` in the probe.
5. **Issue 6** — confirm the model tag.
6. **Issue 3** — remove or relocate `think`, after checking the Ollama version's behaviour.

## How to verify once fixed

```bash
ollama serve                 # in a separate terminal
python main.py --ui          # then run an assessment from Stage 1
```

Expect: the narrative reads as model-written prose rather than the templated fallback sentence, and the grounding banner reports a **non-zero** number of verified claims. If it still says "All 0 clinical claims", Issue 1 is not resolved.
