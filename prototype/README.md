# TriageSense

An AI decision-support layer for ED triage: it recommends, a clinician decides.
This is a real, runnable prototype (not a mockup) — a rules + weighted-scoring
hybrid engine, a Streamlit UI on top of it, 18 simulated patient records, and
an automated test suite that locks in the safety-critical behaviors.

## Run it

```bash
pip install -r requirements.txt

# 1. See it work with no UI (prints everything below in ~1 second):
python simulate_queue.py

# 2. Run the safety-property test suite:
pytest test_scoring_engine.py -v

# 3. Launch the interactive prototype:
streamlit run app.py
```

## Files

| File | What it is |
|---|---|
| `scoring_engine.py` | The actual decision logic. Pure functions, no UI/IO — importable, testable, swappable. |
| `simulate_queue.py` | CLI proof-of-behavior: scores all 18 patients, runs surge mode, checks wait breaches, logs one override, previews de-identified export. |
| `test_scoring_engine.py` | 8 automated tests that encode the safety properties (red flags always win, surge is never less cautious, etc.) |
| `app.py` | Streamlit UI: live queue, patient detail with explainable rationale, override capture, audit log, data-protection tab. |
| `data/patients.csv` | 18 simulated patients covering every required edge case (see below). |
| `config/hospital_config.json` | Every threshold a new hospital would tune — nothing is hard-coded (scalability requirement). |

## How the 18 simulated patients map to the brief's required cases

| Requirement | Patient(s) |
|---|---|
| Ambiguous / overlapping presentation | P07 (vague chest discomfort, high pain but reassuring vitals), P17 (chest tightness read as panic attack) |
| Pediatric case | P03 (2-month infant, fever+lethargy — red flag), P05 (toddler fever) |
| Geriatric case | P01 (stroke), P11 (subtle geriatric sepsis-like presentation), P13 (fall), P14 (zero-history + ambiguous + elderly, compound case) |
| Zero-history / first-time patient | P03, P09, P14, P15 |
| Under-triage-risk red flags | P01 stroke, P02 cardiac, P03 infant lethargy+fever, P04 respiratory distress, P09 hemorrhage |
| Clearly minor (control cases) | P06, P08, P15, P16, P18 |
| Chronic condition flare, returning patient | P10 (COPD) |

## Design decisions worth defending on camera

**1. Rules layer is deterministic and always wins.** Red flags (stroke signs,
crushing chest pain, severe bleeding, unresponsiveness, anaphylaxis, infant
lethargy+fever) force ESI-1 regardless of the composite score. This is not
statistical — it's an explicit safety net, the same pattern used in
[openTriage]'s architecture of
separating rule-based and scored layers.

**2. Age-adjusted vital bands, not one adult model.** `scoring_engine.py`
defines separate normal ranges for infant / child / adult / geriatric, plus a
band-specific fever threshold (37.8°C for geriatric vs 38.3°C for adult,
reflecting blunted fever response in older patients) and an age-risk
multiplier (1.25x infant, 1.2x geriatric) — directly answering the brief's
"a single adult-calibrated model introduces silent safety risk" warning.

**3. High-risk chief-complaint ceiling, independent of vitals.** Chest pain,
breathing trouble, sudden severe headache, syncope, confusion, and a few
others cap the maximum ESI even when vitals look reassuring — because vitals
can look normal early in a serious presentation. This is what makes P07 (chest
discomfort, near-normal vitals) come out at ESI-3, not the ESI-5 a naive
vitals-only model would give it.

**4. Confidence is explicit and asymmetric.** Every recommendation ships with
a confidence percentage built from data completeness, history availability,
and a consistency check (self-reported pain vs. objective vitals). When
confidence falls below threshold, the system can only ever escalate, never
downgrade — enforced by `test_low_confidence_biases_toward_escalation_not_downgrade`.

**5. Surge mode is provably never less cautious.** Toggling surge raises the
confidence threshold and shifts every borderline case one ESI band more
urgent. `test_surge_mode_is_never_less_cautious_than_normal_mode` locks this
in as a regression test, not just a claim.

**6. Waiting-room monitoring.** `check_wait_breach()` re-flags any queued
patient whose wait has exceeded the safe limit for their acuity level —
addresses the brief's requirement to monitor patients already waiting, not
just patients at intake.

**7. Data protection is code, not just a policy paragraph.**
`deidentify_for_training()` one-way-hashes the patient ID before any record
would leave the clinical system for weekly retraining; name/MRN/free-text
never travel with it. The Data Protection tab in the app shows exactly what
that export would look like.

## Assumptions (stated explicitly, per the brief's instructions)

- **Jurisdiction:** US, HIPAA. (`config/hospital_config.json` → `jurisdiction`.)
  Emergency treatment doesn't require prior consent for triage itself; broader
  data use (e.g. research) would need its own consent/IRB pathway, out of
  scope here.
- **Output scale:** ESI 1–5 (Emergency Severity Index), chosen because it's
  the most widely deployed 5-level scale in the US and maps cleanly to a
  single acuity number the queue can sort on.
- **Mixed data availability:** modeled directly — `has_history` is False for
  4 of the 18 patients, and `_completeness()` measures how many of the 7
  expected fields are actually present per patient.
- **Scale:** thresholds in `hospital_config.json` assume a mid-size ED
  (~220 visits/day) but every number is config, not code, so a 100-visit
  rural site or a 500+-visit urban trauma center would just ship a different
  config file, not a different codebase.

## Known limitations (disclosed on purpose — this is a first-version prototype)

- Chief-complaint risk detection is keyword matching. It has one deliberate,
  demonstrated failure mode: it doesn't do negation ("bleeding, now stopped"
  vs. "actively bleeding"), which is exactly why P16 (resolved minor bleeding)
  needed the `bleeding` keyword *removed* from the ceiling list during
  development — real deployment needs proper clinical NLP here.
- The confidence model is a transparent heuristic, not a calibrated
  statistical model. That's a deliberate choice for explainability in a first
  version — the natural next step, once enough logged outcomes accumulate
  through the audit log, is training a real risk model on that data (the same
  path openTriage and the MIMIC-IV-ED benchmarking literature take), with the
  rules layer staying in place as a permanent safety floor underneath it.
- All numeric thresholds (vital bands, ESI cutoffs, confidence weights) are
  illustrative defaults for this prototype and must be clinically validated
  and signed off by a medical director before any real-world use.

## Reference points (ideas adapted, no code copied)

- **ESI v4** (AHRQ) — output scale.
- **NEWS2** — inspiration for 0–3 per-vital deviation banding instead of a
  single opaque score.
  architecture pattern: deterministic rule layer + separately swappable
  scored/model layer.
- MIMIC-IV-ED benchmarking papers / AutoScore literature — inspiration for
  keeping every input's contribution to the score inspectable.
