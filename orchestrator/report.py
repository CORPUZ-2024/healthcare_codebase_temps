"""
report.py — turn a list of RunResult into console, JSON, Markdown and HTML.

Output layout (all under orchestrator/run_status/):
    LATEST.md / latest.json / latest.html      overwritten every run
    history/<YYYY-MM-DDTHH-MM-SS>/             one folder per run
        summary.json  summary.md  summary.html
        logs/<template>.log  logs/<template>.junit.xml
    run_log.csv                                 one appended row per template per run
"""
from __future__ import annotations

import csv
import html
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from runner import RunResult

STATUS_ORDER = ["FAIL", "ERROR", "TIMEOUT", "NO_TESTS", "PASS"]


def totals(results: list[RunResult]) -> dict:
    return {
        "templates": len(results),
        "passed_templates": sum(r.status == "PASS" for r in results),
        "tests_passed": sum(r.passed for r in results),
        "tests_failed": sum(r.failed for r in results),
        "tests_errors": sum(r.errors for r in results),
        "tests_skipped": sum(r.skipped for r in results),
        "tests_xfailed": sum(r.xfailed for r in results),
        "seconds": round(sum(r.seconds for r in results), 2),
        "overall": "PASS" if results and all(r.status == "PASS" for r in results) else "FAIL",
    }


def console_table(results: list[RunResult]) -> str:
    head = f"{'ID':<4} {'TEMPLATE':<40} {'PASS':>5} {'FAIL':>5} {'ERR':>4} {'SKIP':>5} {'SEC':>7}  STATUS"
    rows = [head, "-" * len(head)]
    for r in results:
        rows.append(f"{r.tid:<4} {r.name[4:]:<40} {r.passed:>5} {r.failed:>5} {r.errors:>4} "
                    f"{r.skipped + r.xfailed:>5} {r.seconds:>7.1f}  {r.status}"
                    + (f"  ({r.message})" if r.message and r.status != "PASS" else ""))
    t = totals(results)
    rows.append("-" * len(head))
    rows.append(f"{t['templates']} templates | {t['passed_templates']} green | {t['tests_passed']} tests passed | "
                f"{t['tests_failed']} failed | {t['tests_errors']} errors | {t['seconds']} s | OVERALL {t['overall']}")
    return "\n".join(rows)


def to_markdown(results: list[RunResult], meta: dict) -> str:
    t = totals(results)
    lines = [f"# Orchestrator run — {meta['started']}", "",
             f"**Overall: {t['overall']}** · {t['passed_templates']}/{t['templates']} templates green · "
             f"{t['tests_passed']} tests passed · {t['tests_failed']} failed · {t['tests_errors']} errors · {t['seconds']} s", "",
             f"Mode: `{meta['mode']}` · Python: `{meta['python']}` · Platform: `{meta['platform']}`", "",
             "| ID | Template | Pass | Fail | Err | Skip/xfail | Sec | Status | Note |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for r in results:
        lines.append(f"| {r.tid} | {r.name[4:]} | {r.passed} | {r.failed} | {r.errors} | {r.skipped + r.xfailed} | "
                     f"{r.seconds} | {'✅' if r.status == 'PASS' else '❌'} {r.status} | {r.message.replace('|', '/')} |")
    lines += ["", "Logs for each template are in this run's `logs/` folder "
              f"(`history/{meta['run_id']}/logs/`)."]
    return "\n".join(lines) + "\n"


def to_html(results: list[RunResult], meta: dict) -> str:
    t = totals(results)
    rows = "".join(
        f"<tr class='{r.status.lower()}'><td>{r.tid}</td><td>{html.escape(r.name[4:])}</td><td>{r.passed}</td>"
        f"<td>{r.failed}</td><td>{r.errors}</td><td>{r.skipped + r.xfailed}</td><td>{r.seconds}</td>"
        f"<td><b>{r.status}</b></td><td>{html.escape(r.message)}</td></tr>" for r in results)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Test run {meta['run_id']}</title>
<style>
:root{{--bg:#fff;--ink:#1f2328;--line:#ddd;--ok:#e6f4ea;--bad:#fde7e9;--muted:#666}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15181c;--ink:#e6e8eb;--line:#2c3137;--ok:#173522;--bad:#3d1b1f;--muted:#9aa}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.5 Segoe UI,Arial,sans-serif;margin:24px 16px}}
table{{border-collapse:collapse;width:100%;max-width:1100px}}td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
tr.pass td{{background:var(--ok)}}tr:not(.pass) td{{background:var(--bad)}}.m{{color:var(--muted)}}
.wrap{{overflow-x:auto}}</style></head><body>
<h1>Orchestrator run — {html.escape(meta['started'])}</h1>
<p><b>Overall {t['overall']}</b> · {t['passed_templates']}/{t['templates']} templates green · {t['tests_passed']} tests passed ·
{t['tests_failed']} failed · {t['tests_errors']} errors · {t['seconds']} s</p>
<p class="m">mode {meta['mode']} · python {html.escape(meta['python'])} · {html.escape(meta['platform'])}</p>
<div class="wrap"><table><thead><tr><th>ID</th><th>Template</th><th>Pass</th><th>Fail</th><th>Err</th><th>Skip</th><th>Sec</th><th>Status</th><th>Note</th></tr></thead>
<tbody>{rows}</tbody></table></div></body></html>"""


def write_all(results: list[RunResult], status_root: Path, run_dir: Path, meta: dict) -> dict:
    """Write the run folder, overwrite LATEST.*, append run_log.csv. Returns paths written."""
    status_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "totals": totals(results), "results": [r.to_dict() for r in results]}
    md, page = to_markdown(results, meta), to_html(results, meta)
    for folder in (run_dir,):
        (folder / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (folder / "summary.md").write_text(md, encoding="utf-8")
        (folder / "summary.html").write_text(page, encoding="utf-8")
    (status_root / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (status_root / "LATEST.md").write_text(md, encoding="utf-8")
    (status_root / "latest.html").write_text(page, encoding="utf-8")

    log_csv = status_root / "run_log.csv"
    new = not log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "tid", "template", "mode", "status", "passed", "failed", "errors", "seconds"])
        for r in results:
            w.writerow([meta["run_id"], r.tid, r.name, r.mode, r.status, r.passed, r.failed, r.errors, r.seconds])
    return {"run_dir": str(run_dir), "latest_md": str(status_root / "LATEST.md")}


def run_meta(mode: str, python: str | None) -> dict:
    now = datetime.now()
    return {"run_id": now.strftime("%Y-%m-%dT%H-%M-%S"), "started": now.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode, "python": python or sys.executable,
            "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}"}
