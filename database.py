import json
import os
from datetime import datetime

DB_FILE      = "cases.json"
RECORDS_FILE = "patient_records.json"

# ── Cases ──────────────────────────────────────────────────────
def load_cases():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_case(
    patient_name, patient_age, patient_gender,
    symptoms, quality_score, probs,
    detected_conditions, risk_level,
    image_path, heatmap_path,
    patient_email=""
):
    cases   = load_cases()
    case_id = f"CASE-{len(cases)+1:04d}"
    case    = {
        "case_id":             case_id,
        "timestamp":           datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "status":              "Pending",
        "patient_name":        patient_name,
        "patient_age":         int(patient_age),
        "patient_gender":      patient_gender,
        "symptoms":            symptoms,
        "quality_score":       int(quality_score),
        "probs":               probs.tolist(),
        "detected_conditions": [(n, float(p)) for n, p in detected_conditions],
        "risk_level":          risk_level,
        "image_path":          image_path,
        "heatmap_path":        heatmap_path,
        "patient_email":       patient_email,
        "doctor_diagnosis":    "",
        "doctor_prescription": "",
        "doctor_referral":     "",
        "doctor_notes":        "",
        "reviewed_at":         ""
    }
    cases.append(case)
    with open(DB_FILE, 'w') as f:
        json.dump(cases, f, indent=2)

    # Auto-update patient record
    _register_visit(patient_email, patient_name, patient_age, patient_gender, case_id)
    return case_id

def update_case(case_id, diagnosis, prescription, referral, notes):
    cases = load_cases()
    for case in cases:
        if case["case_id"] == case_id:
            case["doctor_diagnosis"]    = diagnosis
            case["doctor_prescription"] = prescription
            case["doctor_referral"]     = referral
            case["doctor_notes"]        = notes
            case["status"]              = "Reviewed"
            case["reviewed_at"]         = datetime.now().strftime("%d %b %Y, %I:%M %p")
            break
    with open(DB_FILE, 'w') as f:
        json.dump(cases, f, indent=2)

# ── Patient Records ────────────────────────────────────────────
def _load_records():
    if not os.path.exists(RECORDS_FILE):
        return {}
    with open(RECORDS_FILE, 'r') as f:
        return json.load(f)

def _save_records(records):
    with open(RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)

def _register_visit(patient_email, name, age, gender, case_id):
    if not patient_email:
        return
    records = _load_records()
    if patient_email not in records:
        records[patient_email] = {
            "profile": {
                "name":                name,
                "age":                 age,
                "gender":              gender,
                "patient_id":          f"P-{abs(hash(patient_email)) % 99999:05d}",
                "joined":              datetime.now().strftime("%d %b %Y"),
                "blood_group":         "",
                "known_conditions":    [],
                "family_history":      [],
                "current_medications": [],
                "allergies":           []
            },
            "continuity_notes": [],
            "visit_ids":        []
        }
    if case_id not in records[patient_email]["visit_ids"]:
        records[patient_email]["visit_ids"].append(case_id)
    # Keep profile name/age/gender up to date
    records[patient_email]["profile"]["name"]   = name
    records[patient_email]["profile"]["age"]    = age
    records[patient_email]["profile"]["gender"] = gender
    _save_records(records)

def get_patient_record(patient_email):
    records = _load_records()
    return records.get(patient_email)

def update_patient_profile(patient_email, profile_data):
    records = _load_records()
    if patient_email not in records:
        return
    records[patient_email]["profile"].update(profile_data)
    _save_records(records)

def add_continuity_note(patient_email, doctor_name, note):
    records = _load_records()
    if patient_email not in records:
        return
    records[patient_email]["continuity_notes"].append({
        "doctor_name": doctor_name,
        "date":        datetime.now().strftime("%d %b %Y"),
        "note":        note
    })
    _save_records(records)

def get_patient_visits(patient_email):
    all_cases = load_cases()
    visits    = [c for c in all_cases if c.get("patient_email","") == patient_email]
    return sorted(visits, key=lambda c: c["timestamp"])

def get_disease_trend(patient_email, disease_name):
    from constants import DISEASE_NAMES          # ← change this line
    visits  = get_patient_visits(patient_email)
    if disease_name not in DISEASE_NAMES:
        return []
    idx     = DISEASE_NAMES.index(disease_name)
    trend   = []
    for v in visits:
        probs = v.get("probs", [])
        if len(probs) > idx:
            trend.append((v["timestamp"][:6], float(probs[idx])))
    return trend

def get_risk_trend(patient_email):
    visits = get_patient_visits(patient_email)
    def risk_score(r):
        r = r.lower()
        if "high" in r or "specialist" in r:
            return 3
        elif "moderate" in r or "follow" in r:
            return 2
        return 1
    return [(v["timestamp"][:6], risk_score(v["risk_level"])) for v in visits]

def load_appointments():
    APPOINTMENTS_FILE = "appointments.json"
    if not os.path.exists(APPOINTMENTS_FILE):
        return []
    with open(APPOINTMENTS_FILE, 'r') as f:
        return json.load(f)