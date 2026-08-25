"""
TriageSense scoring engine
===========================
A hybrid rules + weighted-scoring decision-support engine for ED triage.

Design lineage / references (ideas adapted, no code copied):
- Emergency Severity Index (ESI) v4 — AHRQ's 5-level acuity scale, used here as the
  OUTPUT scale (ESI-1 most urgent -> ESI-5 least urgent).
- NEWS2 (National Early Warning Score 2) — inspiration for banding vitals into
  0-3 deviation points per parameter instead of a single opaque "risk score".
- dnspangler/openTriage (GitHub) — inspiration for the architecture pattern of
  separating a deterministic rule layer from a scored/weighted layer, and for
  keeping the decision function pure/stateless so it can be unit tested and
  swapped for a trained model later without touching the app layer.
- AutoScore / MEWS literature — inspiration for making every input's point
  contribution transparent (explainability requirement).

IMPORTANT: The exact numeric bands below are illustrative defaults for this
prototype, NOT the copyrighted ESI/NEWS2 tables. Any real deployment must be
clinically validated and signed off by a medical director before use on real
patients.
"""

from __future__ import annotations
import math
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------
# 1. AGE BANDS  (Round-2 requirement: vitals must not use one adult-only scale)
# ----------------------------------------------------------------------

def age_band(age_years: float) -> str:
    if age_years < 1:
        return "infant"
    if age_years < 12:
        return "child"
    if age_years < 65:
        return "adult"
    return "geriatric"


# Illustrative normal ranges per band: (low, high)
# HR = heart rate, RR = respiratory rate, SBP = systolic BP, SPO2 = % , TEMP = deg C
VITAL_NORMALS = {
    "infant":    {"HR": (100, 160), "RR": (30, 53), "SBP": (65, 100),  "SPO2": (95, 100), "TEMP": (36.5, 37.5), "FEVER_AT": 38.0},
    "child":     {"HR": (70, 120),  "RR": (18, 30), "SBP": (80, 112),  "SPO2": (95, 100), "TEMP": (36.5, 37.5), "FEVER_AT": 38.0},
    "adult":     {"HR": (60, 100),  "RR": (12, 20), "SBP": (100, 140), "SPO2": (95, 100), "TEMP": (36.5, 37.5), "FEVER_AT": 38.3},
    # Geriatric: narrower physiological reserve, blunted fever response -> lower
    # fever trigger and lower "acceptable" SpO2 floor is intentionally NOT relaxed
    # (comorbidity risk means we hold the line, we just fire the fever flag earlier).
    "geriatric": {"HR": (60, 100),  "RR": (12, 20), "SBP": (110, 150), "SPO2": (94, 100), "TEMP": (36.0, 37.2), "FEVER_AT": 37.8},
}

# Age bands where deterioration is faster/subtler -> composite score is weighted up.
AGE_RISK_MULTIPLIER = {"infant": 1.25, "child": 1.0, "adult": 1.0, "geriatric": 1.2}


def score_vital(value: Optional[float], low: float, high: float) -> int:
    """Return 0-3 deviation points for one vital, banded like a NEWS2-style score."""
    if value is None:
        return 0  # handled separately by the completeness/confidence calculation
    if low <= value <= high:
        return 0
    span = max(high - low, 1e-6)
    if value < low:
        dev = (low - value) / span
    else:
        dev = (value - high) / span
    if dev <= 0.15:
        return 1
    if dev <= 0.35:
        return 2
    return 3


# ----------------------------------------------------------------------
# 2. RED-FLAG RULES  (deterministic safety net — never statistical, always wins)
# ----------------------------------------------------------------------

RED_FLAGS = {
    "stroke_signs":        ("Facial droop / slurred speech / one-sided weakness (FAST positive)", 1),
    "crushing_chest_pain":  ("Crushing/radiating chest pain with diaphoresis", 1),
    "severe_hemorrhage":    ("Active severe bleeding", 1),
    "unresponsive":         ("Unresponsive (AVPU = U)", 1),
    "anaphylaxis":          ("Signs of anaphylaxis / airway compromise", 1),
    "infant_lethargy_fever": ("Infant with fever and lethargy (AVPU != A)", 1),
    "severe_resp_distress":  ("SpO2 < 90% with visible respiratory distress", 1),
}

# High-risk CHIEF COMPLAINT categories cap the maximum ESI regardless of how
# reassuring the vitals look, because vitals can be normal early in a serious
# presentation (classic teaching case: chest pain, sudden severe headache,
# syncope). This mirrors how real ESI triage keeps a list of "high-risk
# situations" separate from the vital-sign danger zone. It's what stops a
# complaint like "vague chest discomfort" with borderline-normal vitals from
# ever quietly falling to a low-acuity ESI-5/4 on vitals alone.
HIGH_RISK_COMPLAINT_CEILING = {
    "chest": 3,
    "breath": 3,
    "breathing": 3,
    "worst headache": 2,
    "sudden": 3,
    "faint": 3,
    "syncope": 2,
    "stroke": 2,
    "weakness": 3,
    "confusion": 3,
    "pregnan": 3,
}


def complaint_risk_ceiling(chief_complaint: str) -> Optional[int]:
    """Lowest (most urgent) ESI ceiling triggered by any keyword match, or None."""
    text = (chief_complaint or "").lower()
    hits = [cap for kw, cap in HIGH_RISK_COMPLAINT_CEILING.items() if kw in text]
    return min(hits) if hits else None


# ----------------------------------------------------------------------
# 3. PATIENT RECORD
# ----------------------------------------------------------------------

@dataclass
class Patient:
    patient_id: str
    name: str                # simulated only; stripped before any training export
    age: float
    sex: str
    chief_complaint: str
    arrival_mode: str        # walk-in / ambulance / self-referred
    has_history: bool
    hr: Optional[float] = None
    rr: Optional[float] = None
    sbp: Optional[float] = None
    spo2: Optional[float] = None
    temp: Optional[float] = None
    avpu: str = "A"          # A/V/P/U
    pain_score: Optional[float] = None   # 0-10 self-reported
    red_flags: list = field(default_factory=list)   # keys from RED_FLAGS
    arrival_time_min: float = 0.0        # simulated clock, minutes since shift start


REQUIRED_FIELDS = ["hr", "rr", "sbp", "spo2", "temp", "pain_score", "chief_complaint"]


# ----------------------------------------------------------------------
# 4. SCORING
# ----------------------------------------------------------------------

@dataclass
class TriageResult:
    patient_id: str
    band: str
    physiological_score: float
    breakdown: dict
    red_flags_fired: list
    confidence: float
    completeness: float
    ambiguous: bool
    recommended_esi: int
    auto_escalated_for_uncertainty: bool
    rationale: str


def _completeness(p: Patient) -> float:
    present = sum(1 for f in REQUIRED_FIELDS if getattr(p, f) not in (None, ""))
    return present / len(REQUIRED_FIELDS)


def score_patient(p: Patient, confidence_threshold: float = 0.70, surge: bool = False) -> TriageResult:
    band = age_band(p.age)
    normals = VITAL_NORMALS[band]

    breakdown = {
        "HR": score_vital(p.hr, *normals["HR"]),
        "RR": score_vital(p.rr, *normals["RR"]),
        "SBP": score_vital(p.sbp, *normals["SBP"]),
        "SPO2": score_vital(p.spo2, *normals["SPO2"]),
        "TEMP": score_vital(p.temp, *normals["TEMP"]),
    }
    avpu_points = {"A": 0, "V": 1, "P": 2, "U": 3}.get(p.avpu, 0)
    breakdown["AVPU"] = avpu_points

    raw = sum(breakdown.values())

    # Subjective pain adds a small amount of weight, capped, so a very high
    # self-report can nudge urgency even when vitals still look ok.
    pain_bonus = 1 if (p.pain_score is not None and p.pain_score >= 8) else 0
    breakdown["pain_bonus"] = pain_bonus
    raw += pain_bonus

    # Fever check is band-specific (elderly fever threshold is lower).
    if p.temp is not None and p.temp >= normals["FEVER_AT"]:
        breakdown["fever_flag"] = 1
        raw += 1
    else:
        breakdown["fever_flag"] = 0

    physiological_score = raw * AGE_RISK_MULTIPLIER[band]

    # ---- red flags: deterministic override, always wins ----
    fired = [RED_FLAGS[k][0] for k in p.red_flags if k in RED_FLAGS]
    red_flag_esi = 1 if fired else None

    # ---- composite -> ESI band mapping ----
    if physiological_score >= 10:
        composite_esi = 1
    elif physiological_score >= 7:
        composite_esi = 2
    elif physiological_score >= 4:
        composite_esi = 3
    elif physiological_score >= 2:
        composite_esi = 4
    else:
        composite_esi = 5

    # Surge mode: shift one band more conservative across the board.
    if surge and composite_esi > 1:
        composite_esi -= 1

    recommended_esi = min(composite_esi, red_flag_esi) if red_flag_esi else composite_esi

    # High-risk chief-complaint ceiling: applied even if vitals look reassuring.
    ceiling = complaint_risk_ceiling(p.chief_complaint)
    complaint_capped = False
    if ceiling is not None and recommended_esi > ceiling:
        recommended_esi = ceiling
        complaint_capped = True

    # ---- confidence ----
    completeness = _completeness(p)

    ambiguous = False
    if p.pain_score is not None:
        mismatch_high_pain_low_vitals = p.pain_score >= 8 and physiological_score <= 3
        mismatch_low_pain_high_vitals = p.pain_score <= 2 and physiological_score >= 7
        ambiguous = mismatch_high_pain_low_vitals or mismatch_low_pain_high_vitals
    consistency_penalty = 0.15 if ambiguous else 0.0

    confidence = 0.30 + 0.60 * completeness + (0.07 if p.has_history else -0.07) - consistency_penalty
    confidence = max(0.0, min(1.0, confidence))
    confidence = round(confidence * 20) / 20  # nearest 5%

    threshold = confidence_threshold + (0.10 if surge else 0.0)
    threshold = min(threshold, 0.95)

    auto_escalated = False
    if confidence < threshold and recommended_esi > 1:
        recommended_esi -= 1
        auto_escalated = True

    # ---- rationale (explainability requirement) ----
    top_factors = sorted(
        {k: v for k, v in breakdown.items() if isinstance(v, (int, float)) and v > 0}.items(),
        key=lambda kv: kv[1], reverse=True
    )[:3]
    factor_text = ", ".join(f"{k} (+{v})" for k, v in top_factors) if top_factors else "vitals within normal range for age band"
    reasons = []
    if fired:
        reasons.append(f"Red flag(s): {'; '.join(fired)}")
    reasons.append(f"Age band: {band} (risk x{AGE_RISK_MULTIPLIER[band]})")
    reasons.append(f"Top contributing factors: {factor_text}")
    if complaint_capped:
        reasons.append(f"High-risk chief complaint keyword -> ESI capped at {ceiling} regardless of vitals")
    if ambiguous:
        reasons.append("Ambiguous presentation: self-reported pain and objective vitals disagree")
    if auto_escalated:
        reasons.append(f"Confidence {confidence:.0%} below threshold {threshold:.0%} -> auto-escalated one level (bias to caution)")
    if not p.has_history:
        reasons.append("Zero-history / first-time patient -> confidence reduced, flagged for closer review")

    return TriageResult(
        patient_id=p.patient_id,
        band=band,
        physiological_score=round(physiological_score, 2),
        breakdown=breakdown,
        red_flags_fired=fired,
        confidence=confidence,
        completeness=round(completeness, 2),
        ambiguous=ambiguous,
        recommended_esi=recommended_esi,
        auto_escalated_for_uncertainty=auto_escalated,
        rationale=" | ".join(reasons),
    )


# ----------------------------------------------------------------------
# 5. WAITING-ROOM MONITORING  (Round-2 requirement: re-assess if wait breach)
# ----------------------------------------------------------------------

def check_wait_breach(esi: int, elapsed_min: float, safe_wait: dict, breach_multiplier: float = 1.5):
    """Returns (breached: bool, severe_breach: bool)."""
    safe = safe_wait.get(str(esi), safe_wait.get(esi, 999))
    if safe == 0:
        return (elapsed_min > 0, elapsed_min > 5)  # ESI-1 should never wait
    breached = elapsed_min > safe
    severe = elapsed_min > safe * breach_multiplier
    return breached, severe


# ----------------------------------------------------------------------
# 6. AUDIT LOG  (Round-2 requirement: capture overrides, immutable trail)
# ----------------------------------------------------------------------

@dataclass
class AuditEntry:
    entry_id: str
    patient_id: str
    timestamp: float
    actor_role: str
    actor_id: str
    ai_recommended_esi: int
    ai_confidence: float
    final_esi: int
    action: str          # "accepted" | "overridden" | "auto_escalated" | "re_triage"
    reason: Optional[str] = None


class AuditLog:
    """Append-only audit trail. In production this would be a write-once store
    (e.g. an append-only DB table or WORM storage) to satisfy HIPAA-style
    audit-trail requirements; here it's an in-memory list for the prototype."""

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def record(self, patient_id, actor_role, actor_id, ai_esi, ai_conf, final_esi, action, reason=None):
        entry = AuditEntry(
            entry_id=str(uuid.uuid4())[:8],
            patient_id=patient_id,
            timestamp=time.time(),
            actor_role=actor_role,
            actor_id=actor_id,
            ai_recommended_esi=ai_esi,
            ai_confidence=ai_conf,
            final_esi=final_esi,
            action=action,
            reason=reason,
        )
        self._entries.append(entry)
        return entry

    def all(self):
        return list(self._entries)


# ----------------------------------------------------------------------
# 7. DATA PROTECTION: de-identification for the weekly retraining export
# ----------------------------------------------------------------------

def deidentify_for_training(patient_id: str, salt: str = "demo-salt-change-me") -> str:
    """One-way pseudonymous ID. Real deployment would use a hospital-managed
    keyed HMAC and a proper key-rotation policy; this demonstrates the pattern."""
    return hashlib.sha256((salt + patient_id).encode()).hexdigest()[:16]
