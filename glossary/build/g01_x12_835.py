"""
G01 — X12 835 remittance -> paid / denied service lines with CARC adjustment codes
==================================================================================
Copy this whole block into a file (e.g. parse_835.py) and run:  python parse_835.py
Requires: pandas   (pip install pandas)
"""
import pandas as pd


def parse_835(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse an X12 835 (Health Care Claim Payment/Advice) into claim and service-line tables.

    Healthcare context
    ------------------
    The 835 is how payers tell providers what they paid and why they didn't pay the rest. Denial
    analytics, cash posting and underpayment recovery all start from it. Structure (simplified):
    CLP = claim payment, SVC = service line, CAS = adjustment (group code + CARC reason + amount),
    DTM*472 = service date. Segments end with '~', elements are separated by '*', and components by ':'.

    Parameters
    ----------
    text : the raw 835 file content (one interchange).

    Returns
    -------
    (claims, lines)
      claims : claim_id, status_cd, charge_amt, paid_amt, patient_resp_amt, payer_claim_id
      lines  : claim_id, line_no, proc_cd, charge_amt, paid_amt, units, svc_dt,
               adj_group_cd, carc_cd, adj_amt (one row per CAS reason; lines without CAS keep NaN)

    Steps
    -----
    1. Split into segments on '~' and elements on '*' (read the actual delimiters from ISA in
       production: element = ISA[3], component = ISA[104], segment terminator = ISA[105]).
    2. Walk segments: CLP opens a claim, SVC opens a line, CAS/DTM attach to the current line
       (or claim-level CAS to the claim if no line is open).
    3. CAS repeats triplets (reason, amount, quantity) after the group code: expand each.

    Common mistakes
    ---------------
    - Hard-coding '*' and '~' (delimiters are declared in ISA and vary by trading partner).
    - Reading only the first CAS triplet (a CAS segment can carry up to 6 reasons).
    - Treating CLP status 22 (reversal) as a new payment instead of a reversal of an earlier one.
    """
    claims, lines = [], []
    cur_claim, cur_line, line_no = None, None, 0
    for seg in [s.strip() for s in text.split("~") if s.strip()]:                  # step 1
        el = seg.split("*")
        tag = el[0]
        if tag == "CLP":                                                             # step 2
            cur_claim, cur_line, line_no = el[1], None, 0
            claims.append({"claim_id": el[1], "status_cd": el[2], "charge_amt": float(el[3]), "paid_amt": float(el[4]),
                           "patient_resp_amt": float(el[5] or 0), "payer_claim_id": el[7] if len(el) > 7 else None})
        elif tag == "SVC" and cur_claim:
            line_no += 1
            proc = el[1].split(":")
            cur_line = {"claim_id": cur_claim, "line_no": line_no, "proc_cd": proc[1] if len(proc) > 1 else proc[0],
                        "charge_amt": float(el[2]), "paid_amt": float(el[3]), "units": float(el[5]) if len(el) > 5 and el[5] else 1.0,
                        "svc_dt": None, "adj": []}
            lines.append(cur_line)
        elif tag == "DTM" and el[1] == "472" and cur_line is not None:
            cur_line["svc_dt"] = pd.to_datetime(el[2], format="%Y%m%d")
        elif tag == "CAS" and cur_line is not None:                                  # step 3
            for i in range(2, len(el), 3):
                if el[i]:
                    cur_line["adj"].append((el[1], el[i], float(el[i + 1])))
    rows = []
    for ln in lines:
        base = {k: v for k, v in ln.items() if k != "adj"}
        if not ln["adj"]:
            rows.append({**base, "adj_group_cd": None, "carc_cd": None, "adj_amt": float("nan")})
        for g, r, a in ln["adj"]:
            rows.append({**base, "adj_group_cd": g, "carc_cd": r, "adj_amt": a})
    return pd.DataFrame(claims), pd.DataFrame(rows)


def denial_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Adjusted dollars by group code (CO contractual, PR patient responsibility, OA other, PI payer
    initiated) and CARC reason, largest first - the starting point for denial root-cause work."""
    a = lines.dropna(subset=["carc_cd"])
    return (a.groupby(["adj_group_cd", "carc_cd"])["adj_amt"].agg(["count", "sum"]).rename(columns={"sum": "adj_amt"})
             .sort_values("adj_amt", ascending=False).reset_index())


# ---------------------------------------------------------------------------
# Self-test: run `python parse_835.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "ISA*00*          *00*          *ZZ*PAYER          *ZZ*PROVIDER       *250115*1200*^*00501*000000001*0*P*:~"
        "ST*835*0001~BPR*I*150.00*C*ACH~"
        "CLP*CLM001*1*300.00*150.00*20.00*MC*PAY0001~"
        "SVC*HC:99214*200.00*120.00**1~DTM*472*20250105~CAS*CO*45*60.00~CAS*PR*2*20.00~"
        "SVC*HC:83036*100.00*30.00**1~DTM*472*20250105~CAS*CO*45*20.00*1*CO*97*50.00~"
        "CLP*CLM002*4*80.00*0.00*0.00*MC*PAY0002~"
        "SVC*HC:97110*80.00*0.00**2~DTM*472*20250107~CAS*CO*50*80.00~"
        "SE*14*0001~"
    )
    claims, lines = parse_835(sample)
    print(claims, "\n", lines, "\n", denial_summary(lines))
    checks = {
        "2 claims parsed": len(claims) == 2,
        "claim 1 paid 150": claims.loc[claims.claim_id == "CLM001", "paid_amt"].iloc[0] == 150.0,
        "CAS with two triplets expands to 2 rows": (lines.proc_cd == "83036").sum() == 2,
        "denied line CARC 50 (not medically necessary)": lines.loc[lines.claim_id == "CLM002", "carc_cd"].iloc[0] == "50",
        "line charge = paid + adjustments (99214)": abs(200 - 120 - lines.loc[lines.proc_cd == "99214", "adj_amt"].sum()) < 1e-9,
        "service date parsed": lines.svc_dt.notna().all(),
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
