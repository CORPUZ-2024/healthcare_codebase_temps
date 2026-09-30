"""Entry point.

    python run.py              full demo -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from skeleton import checks, data, methods  # noqa: E402
from skeleton.config import Config  # noqa: E402


def demo(cfg: Config) -> None:
    df = data.generate(cfg.n_members, cfg.seed)
    for f in checks.check_unique_key(df, "member_id"):
        print(f)
    print("standard :", methods.mean_with_ci(df["cost_amt"]))
    print("alternative:", methods.mean_with_bootstrap_ci(df["cost_amt"]))


def selftest() -> int:
    df = data.generate(200, 1)
    tests = {
        "generator is reproducible": lambda: data.generate(50, 3).equals(data.generate(50, 3)),
        "CI contains the estimate": lambda: (lambda r: r["lo"] < r["estimate"] < r["hi"])(methods.mean_with_ci(df["cost_amt"])),
        "duplicate key detected": lambda: len(checks.check_unique_key(df.iloc[[0, 0]], "member_id")) == 1,
    }
    ok = True
    for name, fn in tests.items():
        passed = bool(fn())
        ok &= passed
        print(("PASS  " if passed else "FAIL  ") + name)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (demo(Config()) or 0))
