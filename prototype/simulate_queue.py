"""
simulate_queue.py
==================
Runnable, no-UI proof that the scoring engine satisfies every Round-2
minimum-prototype requirement. Run with:  python simulate_queue.py

Prints, in order:
  1. Triage scores + confidence for all 18 simulated patients (normal mode)
  2. Same patients re-scored under a simulated 3x surge (shows conservative shift)
  3. A waiting-room breach check against the hospital's safe-wait config
  4. A worked example of a nurse override, captured in the audit log
  5. A de-identified export preview (data protection requirement)
"""
import csv
import json
from scoring_engine import (
    Patient, score_patient, check_wait_breach, AuditLog, deidentify_for_training
)

def load_patients(path="data/patients.csv"):
    patients = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def num(v):
                return float(v) if v not in (None, "",) else None
            red_flags = [row["red_flags"]] if row["red_flags"] else []
            patients.append(Patient(
                patient_id=row["patient_id"], name=row["name"], age=float(row["age"]),
                sex=row["sex"], chief_complaint=row["chief_complaint"], arrival_mode=row["arrival_mode"],
                has_history=(row["has_history"] == "True"),
                hr=num(row["hr"]), rr=num(row["rr"]), sbp=num(row["sbp"]),
                spo2=num(row["spo2"]), temp=num(row["temp"]), avpu=row["avpu"] or "A",
                pain_score=num(row["pain_score"]), red_flags=red_flags,
                arrival_time_min=float(row["arrival_time_min"]),
            ))
    return patients


def load_config(path="config/hospital_config.json"):
    with open(path) as f:
        return json.load(f)


def print_row(p, r, surge=False):
    flag = " \U0001F6A9 RED FLAG" if r.red_flags_fired else ""
    esc = " \u2B06 auto-escalated (low confidence)" if r.auto_escalated_for_uncertainty else ""
    amb = " \u2753 ambiguous" if r.ambiguous else ""
    print(f"  {p.patient_id:4} {p.name[:22]:22} age {p.age:>5.1f} [{r.band:9}] "
          f"score={r.physiological_score:5.2f}  conf={r.confidence:.0%}  "
          f"-> ESI-{r.recommended_esi}{flag}{esc}{amb}")


def main():
    cfg = load_config()
    patients = load_patients()
    threshold = cfg["confidence_threshold_normal"]

    print("=" * 100)
    print(f"1) NORMAL MODE  ({cfg['hospital_name']}, confidence threshold {threshold:.0%})")
    print("=" * 100)
    normal_results = {}
    for p in patients:
        r = score_patient(p, confidence_threshold=threshold, surge=False)
        normal_results[p.patient_id] = r
        print_row(p, r)

    print()
    print("=" * 100)
    print("2) SURGE MODE  (simulated 3x volume -> thresholds tighten, scoring shifts conservative)")
    print("=" * 100)
    surge_threshold = cfg["confidence_threshold_surge"]
    esi_deltas = []
    for p in patients:
        r_surge = score_patient(p, confidence_threshold=surge_threshold, surge=True)
        r_normal = normal_results[p.patient_id]
        delta = r_normal.recommended_esi - r_surge.recommended_esi
        esi_deltas.append(delta)
        marker = f"  (was ESI-{r_normal.recommended_esi} in normal mode)" if delta else ""
        print_row(p, r_surge, surge=True)
        if marker:
            print(f"       {marker}")
    print(f"\n  Summary: {sum(1 for d in esi_deltas if d>0)}/{len(patients)} patients were escalated to a "
          f"more urgent ESI level purely because of surge conditions.")

    print()
    print("=" * 100)
    print("3) WAITING-ROOM BREACH CHECK  (simulated clock = 75 minutes into shift)")
    print("=" * 100)
    now = 75.0
    safe_wait = cfg["safe_wait_minutes"]
    for p in patients:
        r = normal_results[p.patient_id]
        elapsed = now - p.arrival_time_min
        breached, severe = check_wait_breach(r.recommended_esi, elapsed, safe_wait, cfg["reassessment_breach_multiplier"])
        if breached:
            sev = " *** SEVERE BREACH - IMMEDIATE RE-ASSESSMENT ***" if severe else " -> flagged for re-assessment"
            print(f"  {p.patient_id:4} ESI-{r.recommended_esi}  waited {elapsed:5.1f} min "
                  f"(safe limit {safe_wait[str(r.recommended_esi)]} min){sev}")

    print()
    print("=" * 100)
    print("4) NURSE OVERRIDE EXAMPLE  (captured in the immutable audit log)")
    print("=" * 100)
    log = AuditLog()
    target = next(p for p in patients if p.patient_id == "P07")  # ambiguous chest discomfort
    r = normal_results["P07"]
    print(f"  AI recommendation for {target.patient_id} ({target.name}): ESI-{r.recommended_esi} "
          f"at {r.confidence:.0%} confidence")
    print(f"  Rationale: {r.rationale}")
    entry = log.record(
        patient_id=target.patient_id, actor_role="RN", actor_id="nurse_0042",
        ai_esi=r.recommended_esi, ai_conf=r.confidence, final_esi=r.recommended_esi - 1,
        action="overridden",
        reason="Patient history of anxiety noted, but nurse clinically concerned re: atypical cardiac"
               " presentation in a 45F; escalating one level pending ECG.",
    )
    print(f"  -> Nurse overrides to ESI-{entry.final_esi}. Audit entry logged: {entry}")

    print()
    print("=" * 100)
    print("5) DE-IDENTIFIED EXPORT PREVIEW  (weekly retraining feed / data-protection requirement)")
    print("=" * 100)
    for p in patients[:3]:
        print(f"  raw patient_id={p.patient_id} name='{p.name}'  ->  "
              f"training_id={deidentify_for_training(p.patient_id)}  (name/DOB/MRN dropped)")
    print("  ... (name, MRN, and free-text identifiers are never included in the retraining export)")


if __name__ == "__main__":
    main()
