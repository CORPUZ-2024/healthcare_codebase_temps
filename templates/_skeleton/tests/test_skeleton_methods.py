from skeleton import data, methods


def test_standard_and_alternative_agree_roughly():
    x = data.generate(2_000, 0)["cost_amt"]
    a, b = methods.mean_with_ci(x), methods.mean_with_bootstrap_ci(x)
    assert abs(a["estimate"] - b["estimate"]) < 1e-9
    assert abs(a["lo"] - b["lo"]) / a["estimate"] < 0.05
