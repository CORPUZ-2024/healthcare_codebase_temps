"""
t07 — Value-based contract modeling (all contracts FAKE; shaped like MSSP / Medicaid sub-cap / CM vendor deals).

    python run.py              demo on synthetic data -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vbc_contract_modeling import checks, data, methods  # noqa: E402
from vbc_contract_modeling.config import Config  # noqa: E402

OUT = HERE / "outputs"


def _money(x: float) -> str:
    return f"${x:,.0f}" if x >= 0 else f"-${-x:,.0f}"


def _show(result: dict) -> None:
    for row in result["waterfall"].itertuples():
        print(f"  {row.line:<70s} {_money(row.amount_usd):>16s}")


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    print("NOTE: every contract term here is FAKE. Real MSSP rules: 42 CFR 425 + CMS methodology specifications.\n")
    contracts = data.load_contracts(cfg.contracts_dir)
    found = [f for c in contracts.values() for f in checks.check_contract_fields(c)]

    aco = data.generate_aco_year(cfg.n_beneficiaries, cfg.benchmark_pmpy, cfg.true_savings_pct, cfg.seed,
                                 contracts["FAKE_MSSP_ENHANCED"]["truncation_amt"])
    results = []
    for cid in ("FAKE_MSSP_BASIC_A", "FAKE_MSSP_ENHANCED"):
        r = methods.reconcile_shared_savings(contracts[cid], aco["annual_cost_amt"], cfg.benchmark_pmpy, cfg.quality_score)
        results.append(r)
        print(f"== {cid}: {r['outcome']} (savings rate {r['savings_rate']:.2%}, quality {cfg.quality_score:.2f}) ==")
        _show(r)

    sub = data.generate_subcap_members(cfg.n_subcap_members, contracts["FAKE_MEDICAID_SUBCAP"]["cap_pmpm"], seed=cfg.seed)
    rs = methods.reconcile_subcap(contracts["FAKE_MEDICAID_SUBCAP"], sub)
    results.append(rs)
    print(f"\n== FAKE_MEDICAID_SUBCAP: margin {rs['margin_pct']:.1%} before corridor ==")
    _show(rs)

    cm = data.generate_cm_program(cfg.n_cm_participants, seed=cfg.seed)
    rc = methods.reconcile_fee_upside(contracts["FAKE_CM_FEE_UPSIDE"], cm, cfg.cm_quality_met)
    results.append(rc)
    print(f"\n== FAKE_CM_FEE_UPSIDE: savings rate {rc['savings_rate']:.1%}, quality gate "
          f"{'met' if rc['quality_gate_met'] else 'MISSED'} ==")
    _show(rc)

    enh = contracts["FAKE_MSSP_ENHANCED"]
    grid = methods.scenario_grid(enh, cfg.benchmark_pmpy, cfg.n_beneficiaries, cfg.grid_savings_pct, cfg.grid_quality)
    piv = grid.pivot(index="savings_pct", columns="quality", values="aco_pmpy").round(0)
    print(f"\n== Scenario grid (ENHANCED): ACO $ per beneficiary-year by true savings % x quality ==")
    print(piv.rename(index=lambda s: f"{s:+.0%}").to_string())

    print(f"\n== ALTERNATIVE: Monte Carlo ({cfg.n_sims:,} years of random variation + trend risk, sd {cfg.trend_sd:.1%}) ==")
    mc_rows = []
    for cid in ("FAKE_MSSP_BASIC_A", "FAKE_MSSP_ENHANCED"):
        for s_true in (0.0, cfg.true_savings_pct):
            _, summ = methods.monte_carlo_shared_savings(contracts[cid], cfg.n_beneficiaries, cfg.benchmark_pmpy, s_true,
                                                         cfg.quality_score, cfg.n_sims, cfg.trend_sd, cfg.seed)
            mc_rows.append({"contract_id": cid, "true_savings_pct": s_true, **summ})
    mc = pd.DataFrame(mc_rows)
    print(mc[["contract_id", "true_savings_pct", "p_shared_savings", "p_shared_losses", "expected_aco_amount",
              "aco_p05", "aco_p95"]].round(3).to_string(index=False))
    cv = float(aco["annual_cost_amt"].clip(upper=enh["truncation_amt"]).std() / aco["annual_cost_amt"].clip(upper=enh["truncation_amt"]).mean())
    print(f"random-variation MSR (90%, CV {cv:.2f}, n {cfg.n_beneficiaries:,}): "
          f"{methods.msr_for_confidence(cv, cfg.n_beneficiaries):.2%} (+ trend risk, which is why real MSRs are wider)")

    print("\n== Checks ==")
    found += [f for r in results for f in checks.check_reconciliation_closes(r)]
    found += checks.check_corridors(contracts["FAKE_MEDICAID_SUBCAP"]["corridors"])
    found += checks.check_msr_vs_random_variation(enh["msr_pct"], cfg.n_beneficiaries, cv)
    found += checks.check_target_basis(float(np.average(cm["baseline_pmpm"], weights=cm["member_months"])), cfg.population_pmpm)
    for f in found:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")
    print("all reconciliations close (ANL-015)" if not any(f.check_id == "ANL-015" for f in found) else "")

    pd.concat([r["waterfall"].assign(contract_id=r["contract_id"]) for r in results]).to_csv(OUT / "settlement_waterfalls.csv", index=False)
    grid.to_csv(OUT / "scenario_grid_enhanced.csv", index=False)
    mc.to_csv(OUT / "monte_carlo_summary.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    cfg = Config()
    cs = data.load_contracts(cfg.contracts_dir)
    enh, basic = cs["FAKE_MSSP_ENHANCED"], cs["FAKE_MSSP_BASIC_A"]
    costs = pd.Series([9_500.0] * 1_000)
    r_enh = methods.reconcile_shared_savings(enh, costs, 10_000.0, 0.9)
    r_loss = methods.reconcile_shared_savings(enh, pd.Series([10_500.0] * 1_000), 10_000.0, 0.9)
    r_basic_loss = methods.reconcile_shared_savings(basic, pd.Series([10_500.0] * 1_000), 10_000.0, 0.9)
    r_corr = methods.reconcile_shared_savings(enh, pd.Series([9_900.0] * 1_000), 10_000.0, 0.9)
    tests = {
        "ENHANCED 5% savings, q 0.9: ACO gets 0.75 x 0.9 x $500K = $337,500": lambda: abs(r_enh["aco_amount"] - 337_500) < 1e-6,
        "ENHANCED 5% loss: loss rate 1 - 0.675 = 0.325 floored to 0.40 -> owes $200,000": lambda: abs(r_loss["aco_amount"] + 200_000) < 1e-6,
        "BASIC one-sided: a 5% loss costs the ACO nothing": lambda: r_basic_loss["aco_amount"] == 0.0,
        "1% savings is inside the 2% corridor": lambda: r_corr["outcome"] == "within corridor",
        "every reconciliation closes": lambda: not any(checks.check_reconciliation_closes(r) for r in (r_enh, r_loss, r_corr)),
    }
    ok = True
    for name, fn in tests.items():
        passed = bool(fn())
        ok &= passed
        print(("PASS  " if passed else "FAIL  ") + name)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else (demo(Config()) or 0))
