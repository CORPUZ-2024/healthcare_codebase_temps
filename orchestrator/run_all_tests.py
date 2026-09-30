"""
run_all_tests.py — master test orchestrator for every codebase template.

    python orchestrator/run_all_tests.py                 # run every template's pytest suite
    python orchestrator/run_all_tests.py --list          # show what was discovered
    python orchestrator/run_all_tests.py --only t03,t05  # subset by id
    python orchestrator/run_all_tests.py --group D       # subset by workflow type (A-G)
    python orchestrator/run_all_tests.py --intent VALUE  # subset by intent
    python orchestrator/run_all_tests.py --selftest      # use `run.py --selftest` (no pytest needed)
    python orchestrator/run_all_tests.py --jobs 4        # run templates in parallel
    python orchestrator/run_all_tests.py --python .venv\\Scripts\\python.exe

Results are written to orchestrator/run_status/ (see report.py for the layout).
Exit codes: 0 = all templates PASS, 1 = at least one not PASS, 2 = discovery problem.

Independence contract: this program imports only the Python standard library and
its own three modules. It never imports template code; templates never import it.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from discover import apply_filters, discover  # noqa: E402
from report import console_table, run_meta, write_all  # noqa: E402
from runner import run_template  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run every template's tests and write a status report.")
    p.add_argument("--templates-root", default=str(HERE.parent / "templates"))
    p.add_argument("--status-root", default=str(HERE / "run_status"))
    p.add_argument("--list", action="store_true", help="list discovered templates and exit")
    p.add_argument("--only", help="comma-separated template ids, e.g. t00,t05")
    p.add_argument("--group", help="workflow type letter A-G (from template.yaml)")
    p.add_argument("--intent", help="intent, e.g. MEASURE, VALUE, EXPLAIN")
    p.add_argument("--selftest", action="store_true", help="run `python run.py --selftest` instead of pytest")
    p.add_argument("--python", help="interpreter to run templates with (default: this one)")
    p.add_argument("--jobs", type=int, default=1, help="templates to run in parallel")
    p.add_argument("--timeout", type=int, default=300, help="seconds per template")
    p.add_argument("--fail-fast", action="store_true", help="stop after the first non-PASS template (serial only)")
    p.add_argument("-k", dest="keyword", help="pytest -k expression passed to every template")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        templates = apply_filters(discover(Path(args.templates_root)), args.only, args.group, args.intent)
    except FileNotFoundError as e:
        print(f"[discovery] {e}")
        return 2
    if not templates:
        print("[discovery] no templates matched")
        return 2

    if args.list:
        for t in templates:
            print(f"{t.tid}  {t.name:<42} type={t.group or '-'}  intents={','.join(t.intents) or '-'}")
        return 0

    mode = "selftest" if args.selftest else "pytest"
    meta = run_meta(mode, args.python)
    status_root = Path(args.status_root)
    run_dir = status_root / "history" / meta["run_id"]
    logs = run_dir / "logs"
    extra = ["-k", args.keyword] if args.keyword else []

    print(f"Running {len(templates)} template(s) in {mode} mode -> {run_dir}")

    def one(t):
        r = run_template(t, logs, python=args.python, mode=mode, timeout=args.timeout, extra_args=extra)
        print(f"  {r.tid} {r.status:<8} {r.seconds:>6.1f}s  {t.name}", flush=True)
        return r

    results = []
    if args.jobs > 1 and not args.fail_fast:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(one, templates))
    else:
        for t in templates:
            r = one(t)
            results.append(r)
            if args.fail_fast and r.status != "PASS":
                break

    print()
    print(console_table(results))
    paths = write_all(results, status_root, run_dir, meta)
    print(f"\nStatus written to {paths['latest_md']}  (history: {paths['run_dir']})")
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
