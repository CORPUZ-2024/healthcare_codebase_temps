# Test data — t15 operations lifecycle and prior authorization

## Default: synthetic (no download)
`ops_lifecycle_prior_auth/data.py` → `generate()` returns the referrals **as observed at the data cut** and the **truth**
(every future timestamp), so censoring-aware methods can be scored.

## Your own data
One row per referral with: `referral_dt`, `assessment_dt`, `pa_request_ts`, `pa_priority_cd` (expedited / standard),
`pa_decision_ts`, `pa_decision_cd`, `denial_reason`, `appeal_flag`, `overturn_flag`, `soc_dt`, `retained_90d_flag`,
`payer_cd`. Typical sources: the agency EHR or referral-management system, the payer portal / PA vendor export, and
claims for start of care.

## Public counterpart: payer prior-authorization metrics (CMS-0057-F)
* **Rule:** CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F), published January 2024.
  Fact sheet and rule text: search cms.gov for "CMS-0057-F" (rule text is also in the Federal Register, February 2024).
* **What gets published:** impacted payers post aggregated PA metrics on their public websites annually (the first reports,
  covering 2025, are due March 31, 2026): share of requests approved, denied, and approved after appeal; share approved
  with an extended timeframe; and average decision time for standard and expedited requests.
* **Availability:** there is **no single CMS download**. Metrics are posted payer by payer, and formats vary. **Verify
  what your payers have posted** before planning a benchmark. Collect them into a table with `payer_cd`, priority,
  `pct_approved`, `pct_denied`, `pct_approved_after_appeal` and average hours to compare with `pa_metrics()` output.

Files in this folder are gitignored.
