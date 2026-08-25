"""
TriageSense — interactive prototype
Run with:  streamlit run app.py
"""
import json
import time
import random
import copy

import pandas as pd
import streamlit as st

from scoring_engine import (
    Patient, score_patient, check_wait_breach, AuditLog, deidentify_for_training
)
from simulate_queue import load_patients, load_config

st.set_page_config(page_title="TriageSense", layout="wide", page_icon="🩺")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #17212b;
    --muted: #64727d;
    --line: #dce5e2;
    --teal: #087f73;
    --teal-dark: #07584f;
}

.stApp {
    background: radial-gradient(circle at 94% 3%, rgba(8, 127, 115, .10), transparent 26rem),
                linear-gradient(135deg, #f6f8f7 0%, #edf3f0 52%, #f8f7f2 100%);
    color: var(--ink);
    font-family: 'Manrope', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123f3a 0%, #0d2f2c 100%);
    border-right: 1px solid rgba(255,255,255,.10);
}
[data-testid="stSidebar"] * { color: #e8f4f0; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSlider"] {
    background: rgba(255,255,255,.08);
    border-color: rgba(255,255,255,.18);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
.block-container { max-width: 1500px; padding: 3.3rem 3.5rem 4rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
h1 { font-weight: 800; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,.78);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.05rem 1.2rem;
    box-shadow: 0 8px 24px rgba(27, 62, 57, .06);
    transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(27, 62, 57, .12); }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="stMetricValue"] { color: var(--teal-dark); font-size: 2rem; font-weight: 800; }
[data-testid="stTabs"] button { color: var(--muted); font-weight: 700; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--teal); height: 3px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: 0 10px 28px rgba(27, 62, 57, .06); }
[data-testid="stAlert"] { border-radius: 8px; }
button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
button[kind="primary"]:hover { background: var(--teal-dark); border-color: var(--teal-dark); }
.queue-hero {
    display: flex; justify-content: space-between; align-items: flex-end; gap: 2rem;
    margin: .2rem 0 1.6rem; animation: rise-in .55s ease both;
}
.queue-kicker { color: var(--teal); font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .45rem; }
.queue-hero h1 { font-size: clamp(1.9rem, 3vw, 2.8rem); line-height: 1; margin: 0; white-space: nowrap; }
.queue-hero p { color: var(--muted); margin: .7rem 0 0; max-width: 40rem; }
.live-pill {
    display: flex; align-items: center; gap: .55rem; border: 1px solid rgba(8,127,115,.22);
    background: rgba(255,255,255,.72); color: var(--teal-dark); border-radius: 999px;
    padding: .6rem .85rem; font-family: 'DM Mono', monospace; font-size: .72rem; white-space: nowrap;
}
.live-dot { width: .55rem; height: .55rem; border-radius: 50%; background: #31ad72; box-shadow: 0 0 0 5px rgba(49,173,114,.14); animation: pulse 1.8s infinite; }
.status-rail { display: flex; flex-wrap: wrap; gap: .55rem; margin: 0 0 1.5rem; animation: rise-in .7s .08s ease both; }
.status-chip { background: rgba(255,255,255,.62); border: 1px solid var(--line); border-radius: 999px; color: var(--muted); padding: .38rem .72rem; font-size: .72rem; }
.status-chip strong { color: var(--ink); font-family: 'DM Mono', monospace; font-weight: 500; }
@keyframes rise-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }
@media (max-width: 800px) {
    .block-container { padding: 2.5rem 1rem 3rem; }
    .queue-hero { align-items: flex-start; flex-direction: column; gap: 1rem; }
    .queue-hero h1 { font-size: 2.15rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- state ----
if "audit_log" not in st.session_state:
    st.session_state.audit_log = AuditLog()
if "patients" not in st.session_state:
    st.session_state.patients = load_patients()
if "overridden" not in st.session_state:
    st.session_state.overridden = {}   # patient_id -> final_esi
if "clock" not in st.session_state:
    st.session_state.clock = 75.0

cfg = load_config()

# ------------------------------------------------------------- sidebar -----
st.sidebar.title("🩺 TriageSense")
st.sidebar.caption(cfg["hospital_name"] + f"  ·  jurisdiction: {cfg['jurisdiction']}")

role = st.sidebar.selectbox("Logged in as", ["Nurse (RN)", "Admin / Medical director"])
is_admin = role.startswith("Admin")

st.sidebar.divider()
surge = st.sidebar.toggle("🚨 Simulate 3x surge", value=False,
                           help="Tightens the confidence threshold and shifts every ESI band one step more conservative — the system gets MORE cautious under load, not less.")
st.session_state.clock = st.sidebar.slider("Simulated shift clock (minutes elapsed)", 0, 180, int(st.session_state.clock), step=5)

st.sidebar.divider()
st.sidebar.subheader("Hospital configuration")
st.sidebar.caption("Editable only by Admin — every threshold is per-hospital, not hard-coded (scalability requirement).")
conf_threshold = st.sidebar.slider("Confidence threshold (normal)", 0.5, 0.95,
                                    cfg["confidence_threshold_normal"], 0.05, disabled=not is_admin)
if is_admin:
    cfg["confidence_threshold_normal"] = conf_threshold

st.sidebar.divider()
if st.sidebar.button("↻ Reset demo data"):
    st.session_state.patients = load_patients()
    st.session_state.overridden = {}
    st.session_state.audit_log = AuditLog()
    st.rerun()

# ----------------------------------------------------------- scoring ------
threshold = cfg["confidence_threshold_surge"] if surge else cfg["confidence_threshold_normal"]

def compute_all(patients):
    rows = []
    for p in patients:
        r = score_patient(p, confidence_threshold=threshold, surge=surge)
        final_esi = st.session_state.overridden.get(p.patient_id, r.recommended_esi)
        elapsed = st.session_state.clock - p.arrival_time_min
        breached, severe = check_wait_breach(final_esi, elapsed, cfg["safe_wait_minutes"],
                                              cfg["reassessment_breach_multiplier"])
        rows.append(dict(patient=p, result=r, final_esi=final_esi,
                          elapsed=elapsed, breached=breached, severe=severe))
    return rows

rows = compute_all(st.session_state.patients)
rows.sort(key=lambda x: (x["final_esi"], -x["elapsed"]))

# ------------------------------------------------------------- header -----
st.markdown(f"""
<section class="queue-hero">
    <div>
        <div class="queue-kicker">Emergency department / command view</div>
        <h1>TriageSense</h1>
        <p>Live acuity orchestration for the next clinical decision.</p>
    </div>
    <div class="live-pill"><span class="live-dot"></span> LIVE QUEUE · {"SURGE MODE" if surge else "NORMAL MODE"}</div>
</section>
<div class="status-rail">
    <span class="status-chip">SHIFT CLOCK <strong>{st.session_state.clock:.0f} min</strong></span>
    <span class="status-chip">THRESHOLD <strong>{threshold:.0%}</strong></span>
    <span class="status-chip">REASSESSMENT <strong>{sum(1 for r in rows if r["breached"])} flagged</strong></span>
    <span class="status-chip">AUDIT EVENTS <strong>{len(st.session_state.audit_log.all())}</strong></span>
</div>
""", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients in queue", len(rows))
c2.metric("ESI-1 / ESI-2 (critical + urgent)", sum(1 for r in rows if r["final_esi"] <= 2))
c3.metric("Wait breaches", sum(1 for r in rows if r["breached"]))
c4.metric("Mode", "🚨 SURGE (3x)" if surge else "Normal", delta=None)

if surge:
    n_up = sum(1 for r in rows if r["result"].recommended_esi < r["result"].physiological_score)  # placeholder unused
    st.warning(f"Surge mode is active: confidence threshold raised to {threshold:.0%} and every borderline "
               f"case is shifted one ESI band more urgent. This models the safety principle that the system "
               f"should get MORE conservative exactly when clinician bandwidth is thinnest.")

tab_queue, tab_detail, tab_audit, tab_privacy, tab_about = st.tabs(
    ["📋 Queue", "🔍 Patient detail & override", "🧾 Audit log", "🔒 Data protection", "ℹ️ About this prototype"]
)

# --------------------------------------------------------------- queue ----
with tab_queue:
    st.caption("Sorted by final ESI (most urgent first), then by longest wait. "
               "Rows in red have breached the safe-wait threshold for their acuity and need re-assessment now.")
    esi_color = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "⚪"}
    table_rows = []
    for row in rows:
        p, r = row["patient"], row["result"]
        flags = []
        if r.red_flags_fired:
            flags.append("🚩 red flag")
        if r.auto_escalated_for_uncertainty:
            flags.append("⬆ low confidence")
        if r.ambiguous:
            flags.append("❓ ambiguous")
        if row["severe"]:
            flags.append("⛔ SEVERE BREACH")
        elif row["breached"]:
            flags.append("⏱ breach")
        if row["patient"].patient_id in st.session_state.overridden:
            flags.append("✍️ overridden")
        table_rows.append({
            "": esi_color[row["final_esi"]],
            "ID": p.patient_id,
            "Name": p.name,
            "Age": f"{p.age:g} ({r.band})",
            "Complaint": p.chief_complaint,
            "ESI (final)": row["final_esi"],
            "AI confidence": f"{r.confidence:.0%}",
            "Waited (min)": f"{row['elapsed']:.0f}",
            "Flags": " · ".join(flags) if flags else "—",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=560)

# -------------------------------------------------------------- detail ----
with tab_detail:
    ids = [row["patient"].patient_id + " — " + row["patient"].name for row in rows]
    choice = st.selectbox("Select a patient", ids)
    pid = choice.split(" — ")[0]
    row = next(r for r in rows if r["patient"].patient_id == pid)
    p, r = row["patient"], row["result"]

    colA, colB = st.columns([1.3, 1])
    with colA:
        st.subheader(f"{p.name}  ·  age {p.age:g}  ·  {r.band} band")
        st.write(f"**Chief complaint:** {p.chief_complaint}")
        st.write(f"**Arrival mode:** {p.arrival_mode}  ·  **Has prior history on file:** {'Yes' if p.has_history else 'No — first-time / zero-history patient'}")
        vit_cols = st.columns(6)
        for col, (label, val, unit) in zip(vit_cols, [
            ("HR", p.hr, "bpm"), ("RR", p.rr, "/min"), ("SBP", p.sbp, "mmHg"),
            ("SpO2", p.spo2, "%"), ("Temp", p.temp, "°C"), ("Pain", p.pain_score, "/10"),
        ]):
            col.metric(label, f"{val:g}{unit}" if val is not None else "— missing")

        st.markdown("**Per-vital deviation points** (0 = normal, 3 = severe, age-band adjusted)")
        st.bar_chart(pd.Series({k: v for k, v in r.breakdown.items() if k in ["HR", "RR", "SBP", "SPO2", "TEMP", "AVPU"]}))

        st.info(f"**AI rationale:** {r.rationale}")

    with colB:
        st.metric("AI-recommended ESI", f"ESI-{r.recommended_esi}")
        st.metric("Confidence", f"{r.confidence:.0%}", delta=f"{'below' if r.confidence < threshold else 'above'} {threshold:.0%} threshold")
        st.metric("Data completeness", f"{r.completeness:.0%}")
        if row["breached"]:
            st.error(f"⏱ Waited {row['elapsed']:.0f} min — safe limit for ESI-{row['final_esi']} is "
                      f"{cfg['safe_wait_minutes'][str(row['final_esi'])]} min. Flagged for re-assessment.")

        st.divider()
        st.markdown("### Nurse decision")
        if not is_admin:
            options = [1, 2, 3, 4, 5]
            decision = st.radio("Confirm or override the ESI level:", options,
                                 index=options.index(row["final_esi"]), horizontal=True, key=f"radio_{pid}")
            reason = st.text_input("Reason (required if overriding)", key=f"reason_{pid}")
            if st.button("Submit decision", key=f"submit_{pid}"):
                action = "accepted" if decision == r.recommended_esi else "overridden"
                if action == "overridden" and not reason:
                    st.error("A reason is required to log an override.")
                else:
                    st.session_state.overridden[pid] = decision
                    st.session_state.audit_log.record(
                        patient_id=pid, actor_role="RN", actor_id="demo_nurse",
                        ai_esi=r.recommended_esi, ai_conf=r.confidence, final_esi=decision,
                        action=action, reason=reason or "Confirmed AI recommendation",
                    )
                    st.success(f"Logged: {action} → ESI-{decision}")
                    st.rerun()
        else:
            st.caption("Switch role to Nurse (RN) in the sidebar to confirm or override a recommendation.")

# ---------------------------------------------------------------- audit ---
with tab_audit:
    st.caption("Append-only. Every AI recommendation, nurse confirmation, and override is captured here with "
               "who/when/why — this is what a compliance audit or a liability review would pull.")
    entries = st.session_state.audit_log.all()
    if entries:
        df = pd.DataFrame([{
            "Time": time.strftime("%H:%M:%S", time.localtime(e.timestamp)),
            "Patient": e.patient_id, "Actor": f"{e.actor_role} ({e.actor_id})",
            "AI ESI": e.ai_recommended_esi, "AI confidence": f"{e.ai_confidence:.0%}",
            "Final ESI": e.final_esi, "Action": e.action, "Reason": e.reason,
        } for e in entries])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Export audit log (CSV)", df.to_csv(index=False), "audit_log.csv")
    else:
        st.info("No decisions logged yet in this session — confirm or override a patient in the Patient detail tab.")

# -------------------------------------------------------------- privacy ---
with tab_privacy:
    st.markdown(f"""
**Assumed jurisdiction:** {cfg['jurisdiction']} — this affects the audit trail, retention policy,
and consent model below. A real deployment must confirm this with hospital counsel.

- **Consent model:** emergency triage falls under treatment-without-prior-consent provisions for
  emergency care; no separate consent gate blocks triage itself. Downstream use of data (e.g. research)
  would need its own consent/IRB pathway — out of scope for this prototype.
- **Access control:** role-based. Nurses see and act on the live queue; only Admin/medical-director
  roles can edit hospital-wide thresholds. (Demonstrated by the role switch in the sidebar.)
- **Retention:** raw vitals retained {cfg['data_retention']['raw_vitals_days']} days;
  audit log retained {cfg['data_retention']['audit_log_years']} years (liability requirement);
  weekly retraining export is **{cfg['data_retention']['training_export']}**.
- **De-identification for retraining** — preview of what would actually leave the clinical system:
""")
    prev = []
    for p in st.session_state.patients[:6]:
        prev.append({"Raw ID": p.patient_id, "Raw name": p.name,
                     "→ Training ID (irreversible hash)": deidentify_for_training(p.patient_id)})
    st.dataframe(pd.DataFrame(prev), use_container_width=True, hide_index=True)
    st.caption("Name, MRN, and free-text identifiers are dropped entirely before any record reaches the "
               "weekly retraining pipeline — only the hashed ID, clinical inputs, AI recommendation, and "
               "final outcome travel with it.")

# ---------------------------------------------------------------- about ---
with tab_about:
    st.markdown("""
### What this prototype demonstrates
- **Age-adjusted scoring** — infant / child / adult / geriatric vital bands, not one adult-calibrated model.
- **Deterministic red-flag safety net** — stroke signs, crushing chest pain, severe bleeding, unresponsiveness,
  anaphylaxis, infant lethargy+fever always force ESI-1, independent of the composite score.
- **High-risk complaint ceiling** — chest pain, breathing trouble, sudden severe headache etc. can never
  quietly fall to a low-acuity ESI on reassuring vitals alone.
- **Explicit confidence** — every recommendation ships with a confidence value; low confidence
  auto-escalates (never downgrades) — directly implements "bias toward escalation under uncertainty."
- **Surge behavior** — a 3x-load toggle that provably never recommends a *less* urgent ESI than normal mode
  for the same patient (see `test_surge_mode_is_never_less_cautious_than_normal_mode`).
- **Waiting-room monitoring** — re-assessment flags when a patient's wait exceeds the safe limit for their
  acuity level.
- **Audit trail + override capture** — every AI recommendation and every nurse decision is logged.
- **De-identified training export** — patient data is pseudonymized before it ever reaches the retraining loop.

### Reference points this design borrows ideas from (no code copied)
- **ESI (Emergency Severity Index) v4** — the 5-level output scale.
- **NEWS2** — inspiration for banding each vital into 0–3 deviation points instead of a black-box score.
- Public ED-triage benchmarking work (MIMIC-IV-ED, AutoScore literature) — inspiration for keeping every
  input's contribution to the score inspectable (explainability).

### Known limitations (intentionally disclosed)
- Chief-complaint risk detection is keyword-based for this prototype; a production system would need
  proper clinical NLP with negation handling (e.g. "bleeding, now stopped" vs. "actively bleeding").
- Vital-band thresholds are illustrative defaults and **must be clinically validated** by a medical
  director before any real deployment.
- The confidence model is a transparent heuristic, not a calibrated statistical model — chosen
  deliberately for explainability in a first version; a hybrid ML layer (as in openTriage) is the
  natural next step once enough logged outcomes exist to train and validate one safely.
""")
