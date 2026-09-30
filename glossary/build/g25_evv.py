"""
G25 — Electronic Visit Verification (EVV) compliance and authorized-vs-delivered hours
======================================================================================
Copy this whole block into a file (e.g. evv.py) and run:  python evv.py
Requires: pandas, numpy   (pip install pandas numpy)
State EVV rules and aggregator formats differ; thresholds below are illustrative.
"""
import numpy as np
import pandas as pd

REQUIRED = ["service_cd", "member_id", "visit_dt", "worker_id", "check_in_ts", "check_out_ts", "gps_lat", "gps_lon"]


def _km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def evv_compliance(visits: pd.DataFrame, members: pd.DataFrame, max_km: float = 0.5) -> pd.DataFrame:
    """Flag each visit: all required EVV elements present, electronically captured (not manually edited),
    and located at the member's home.

    Healthcare context
    ------------------
    The 21st Century Cures Act requires state Medicaid programs to use EVV for personal care and home
    health visits, capturing: type of service, who received it, date, location, who provided it, and
    start and end times. Agencies are paid (and audited) on compliant visits; manual edits and missing
    GPS are the usual failure points.

    Returns visits + missing_elements_flag, manual_edit_flag, distance_km, location_ok_flag, compliant_flag.
    """
    v = visits.merge(members[["member_id", "home_lat", "home_lon"]], on="member_id", how="left")
    v["missing_elements_flag"] = v[REQUIRED].isna().any(axis=1).astype(int)
    v["manual_edit_flag"] = v["manual_edit_flag"].fillna(0).astype(int)
    v["distance_km"] = _km(v.gps_lat, v.gps_lon, v.home_lat, v.home_lon)
    v["location_ok_flag"] = (v.distance_km <= max_km).astype(int)
    v["compliant_flag"] = ((v.missing_elements_flag == 0) & (v.manual_edit_flag == 0) & (v.location_ok_flag == 1)).astype(int)
    return v


def authorized_vs_delivered(visits: pd.DataFrame, auths: pd.DataFrame, unit_minutes: int = 15,
                            under: float = 0.90) -> pd.DataFrame:
    """Delivered vs. authorized units per member-week: under-delivery (a care gap) and over-delivery (unbillable).

    Parameters
    ----------
    visits : member_id, visit_dt, check_in_ts, check_out_ts (compliant visits only)
    auths : member_id, week_start (Monday), authorized_units (15-minute units)

    Steps
    -----
    1. Units per visit = floor(minutes / unit_minutes) (partial units are not billable).
    2. Sum by member x ISO week; join authorizations.
    3. delivery_ratio = delivered / authorized; flags: under (< ``under``) and over (> 1).

    Common mistakes
    ---------------
    - Rounding partial units up.
    - Measuring against the authorization for the wrong week (weeks start on the plan's defined day).
    - Counting non-compliant EVV visits as delivered (they may be unbillable).
    """
    v = visits.copy()
    minutes = (v.check_out_ts - v.check_in_ts).dt.total_seconds() / 60
    v["units"] = np.floor(minutes / unit_minutes)                                                # step 1
    v["week_start"] = v.visit_dt - pd.to_timedelta(v.visit_dt.dt.weekday, unit="D")
    d = v.groupby(["member_id", "week_start"], as_index=False).units.sum().rename(columns={"units": "delivered_units"})
    out = auths.merge(d, on=["member_id", "week_start"], how="left").fillna({"delivered_units": 0})   # step 2
    out["delivery_ratio"] = out.delivered_units / out.authorized_units                             # step 3
    out["under_delivery_flag"] = (out.delivery_ratio < under).astype(int)
    out["over_delivery_flag"] = (out.delivery_ratio > 1).astype(int)
    return out


# ---------------------------------------------------------------------------
# Self-test: run `python evv.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ts = pd.Timestamp
    members = pd.DataFrame({"member_id": ["A", "B"], "home_lat": [42.3601, 41.8781], "home_lon": [-71.0589, -87.6298]})
    visits = pd.DataFrame({
        "service_cd": ["T1019"] * 4, "member_id": ["A", "A", "B", "B"], "worker_id": ["W1", "W1", "W2", None],
        "visit_dt": pd.to_datetime(["2025-03-03", "2025-03-05", "2025-03-04", "2025-03-06"]),
        "check_in_ts": [ts("2025-03-03 09:00"), ts("2025-03-05 09:00"), ts("2025-03-04 13:00"), ts("2025-03-06 13:00")],
        "check_out_ts": [ts("2025-03-03 11:07"), ts("2025-03-05 11:00"), ts("2025-03-04 15:00"), ts("2025-03-06 15:00")],
        "gps_lat": [42.3602, 42.3601, 41.95, 41.8781], "gps_lon": [-71.0590, -71.0589, -87.6298, -87.6298],
        "manual_edit_flag": [0, 1, 0, 0]})
    c = evv_compliance(visits, members)
    auths = pd.DataFrame({"member_id": ["A", "B"], "week_start": pd.to_datetime(["2025-03-03", "2025-03-03"]),
                          "authorized_units": [16, 16]})
    d = authorized_vs_delivered(c[c.compliant_flag == 1], auths).set_index("member_id")
    print(c[["member_id", "visit_dt", "missing_elements_flag", "manual_edit_flag", "distance_km", "compliant_flag"]].round(2), "\n", d)
    checks = {
        "visit 1 compliant (at home, electronic)": c.compliant_flag[0] == 1,
        "manual edit -> non-compliant": c.compliant_flag[1] == 0,
        "GPS 8 km from home -> non-compliant": c.compliant_flag[2] == 0,
        "missing worker id -> non-compliant": c.compliant_flag[3] == 0,
        "A: 127 minutes -> 8 billable units of 16 authorized (50%, under-delivery)": d.loc["A", ["delivered_units", "under_delivery_flag"]].tolist() == [8, 1],
        "B: no compliant visits -> 0 delivered": d.loc["B", "delivered_units"] == 0,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
