"""
G02 — FHIR R4 bundle flattening (Patient, Encounter, Condition, Observation -> tables)
======================================================================================
Copy this whole block into a file (e.g. fhir_flatten.py) and run:  python fhir_flatten.py
Requires: pandas   (pip install pandas)   Test data: Synthea (synthetichealth.github.io) exports FHIR R4 bundles.
"""
import json

import pandas as pd


def _ref_id(ref: str | None) -> str | None:
    """'Patient/123' or 'urn:uuid:123' -> '123'."""
    if not ref:
        return None
    return ref.split("/")[-1].replace("urn:uuid:", "")


def _coding(cc: dict | None, system_hint: str = "") -> tuple[str | None, str | None]:
    """First coding (preferring a system containing ``system_hint``) -> (code, display)."""
    codings = (cc or {}).get("coding", [])
    pick = next((c for c in codings if system_hint and system_hint in c.get("system", "")), codings[0] if codings else {})
    return pick.get("code"), pick.get("display")


def flatten_bundle(bundle: dict) -> dict[str, pd.DataFrame]:
    """Flatten a FHIR R4 Bundle into one analysis table per resource type.

    Healthcare context
    ------------------
    EHR and HIE feeds increasingly arrive as FHIR (the CMS interoperability rules mandate FHIR APIs).
    Analysts need rows and columns: one table per resource, references reduced to ids, codings reduced
    to (code, system-specific display). Synthea produces realistic R4 bundles to practice on.

    Returns {'patient', 'encounter', 'condition', 'observation'} DataFrames.

    Steps
    -----
    1. Iterate bundle['entry'][*]['resource']; route by resourceType.
    2. Reduce references ('Patient/x', 'urn:uuid:x') to bare ids so tables join.
    3. Pick the coding from the system you need (SNOMED for conditions, LOINC for observations),
       not just the first coding.
    4. Observations: valueQuantity -> value + unit; valueCodeableConcept -> code.

    Common mistakes
    ---------------
    - Taking coding[0] blindly (the same concept may be coded in several systems).
    - Dropping 'urn:uuid:' references (transaction bundles use them; ids won't join).
    - Ignoring Observation components (blood pressure has systolic and diastolic as components).
    """
    out = {"patient": [], "encounter": [], "condition": [], "observation": []}
    for entry in bundle.get("entry", []):                                        # step 1
        r = entry.get("resource", {})
        t = r.get("resourceType")
        if t == "Patient":
            out["patient"].append({"patient_id": r["id"], "birth_dt": r.get("birthDate"), "sex_cd": r.get("gender"),
                                   "zip_cd": (r.get("address") or [{}])[0].get("postalCode")})
        elif t == "Encounter":
            code, disp = _coding((r.get("type") or [{}])[0])
            out["encounter"].append({"encounter_id": r["id"], "patient_id": _ref_id(r.get("subject", {}).get("reference")),  # step 2
                                     "class_cd": r.get("class", {}).get("code"), "type_cd": code, "type_desc": disp,
                                     "start_ts": r.get("period", {}).get("start"), "end_ts": r.get("period", {}).get("end")})
        elif t == "Condition":
            code, disp = _coding(r.get("code"), "snomed")                          # step 3
            out["condition"].append({"condition_id": r["id"], "patient_id": _ref_id(r.get("subject", {}).get("reference")),
                                     "encounter_id": _ref_id(r.get("encounter", {}).get("reference")), "snomed_cd": code,
                                     "condition_desc": disp, "onset_ts": r.get("onsetDateTime")})
        elif t == "Observation":
            base = {"observation_id": r["id"], "patient_id": _ref_id(r.get("subject", {}).get("reference")),
                    "effective_ts": r.get("effectiveDateTime")}
            parts = r.get("component") or [r]                                      # step 4
            for comp in parts:
                code, disp = _coding(comp.get("code"), "loinc")
                vq = comp.get("valueQuantity") or {}
                out["observation"].append({**base, "loinc_cd": code, "obs_desc": disp, "value_num": vq.get("value"),
                                           "unit": vq.get("unit"), "value_cd": _coding(comp.get("valueCodeableConcept"))[0]})
    return {k: pd.DataFrame(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Self-test: run `python fhir_flatten.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bundle = json.loads("""{"resourceType": "Bundle", "type": "transaction", "entry": [
      {"resource": {"resourceType": "Patient", "id": "p1", "birthDate": "1948-03-02", "gender": "female",
                    "address": [{"postalCode": "02139"}]}},
      {"resource": {"resourceType": "Encounter", "id": "e1", "subject": {"reference": "urn:uuid:p1"},
                    "class": {"code": "AMB"}, "type": [{"coding": [{"system": "http://snomed.info/sct", "code": "185349003",
                    "display": "Encounter for check up"}]}], "period": {"start": "2025-01-05T09:00:00Z", "end": "2025-01-05T09:30:00Z"}}},
      {"resource": {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/p1"}, "encounter": {"reference": "Encounter/e1"},
                    "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "E11.9"},
                                        {"system": "http://snomed.info/sct", "code": "44054006", "display": "Diabetes mellitus type 2"}]},
                    "onsetDateTime": "2019-06-01"}},
      {"resource": {"resourceType": "Observation", "id": "o1", "subject": {"reference": "Patient/p1"},
                    "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
                    "valueQuantity": {"value": 7.4, "unit": "%"}, "effectiveDateTime": "2025-01-05"}},
      {"resource": {"resourceType": "Observation", "id": "o2", "subject": {"reference": "Patient/p1"}, "effectiveDateTime": "2025-01-05",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure panel"}]},
                    "component": [
                      {"code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic"}]}, "valueQuantity": {"value": 138, "unit": "mm[Hg]"}},
                      {"code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic"}]}, "valueQuantity": {"value": 84, "unit": "mm[Hg]"}}]}}
    ]}""")
    t = flatten_bundle(bundle)
    for k, v in t.items():
        print(k, "\n", v, "\n")
    checks = {
        "urn:uuid reference resolved to p1": t["encounter"].patient_id.iloc[0] == "p1",
        "SNOMED preferred over ICD-10 coding": t["condition"].snomed_cd.iloc[0] == "44054006",
        "A1c value parsed": t["observation"].query("loinc_cd == '4548-4'").value_num.iloc[0] == 7.4,
        "BP components -> 2 rows": set(t["observation"].query("observation_id == 'o2'").loinc_cd) == {"8480-6", "8462-4"},
        "tables join on patient_id": set(t["condition"].patient_id) <= set(t["patient"].patient_id),
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
