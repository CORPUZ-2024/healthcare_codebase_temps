"""
runner.py — run ONE template's tests in its own subprocess and summarize.

Isolation rules (see orchestrator/README.md):
  * cwd = the template folder, so the template's own pytest.ini applies
  * a fresh Python process per template; nothing is imported into this process
  * results come back only through the exit code and a JUnit XML file
"""
from __future__ import annotations

import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

from discover import TemplateInfo


@dataclass
class RunResult:
    tid: str
    name: str
    mode: str                 # "pytest" or "selftest"
    status: str               # PASS | FAIL | ERROR | TIMEOUT | NO_TESTS
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    seconds: float = 0.0
    exit_code: int | None = None
    log_file: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def parse_junit(xml_path: Path) -> dict:
    """Count outcomes in a pytest JUnit XML file.

    pytest writes xfail as <skipped type="pytest.xfail">; we count those separately
    so an intentional expected-failure is never confused with a skip.
    """
    counts = {"tests": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0}
    root = ET.parse(xml_path).getroot()
    for case in root.iter("testcase"):
        counts["tests"] += 1
        if case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        else:
            sk = case.find("skipped")
            if sk is not None:
                if "xfail" in (sk.get("type", "") + sk.get("message", "")):
                    counts["xfailed"] += 1
                else:
                    counts["skipped"] += 1
    return counts


def run_template(t: TemplateInfo, out_dir: Path, python: str | None = None,
                 mode: str = "pytest", timeout: int = 300, extra_args: list[str] | None = None) -> RunResult:
    """Execute one template's test suite and return a RunResult.

    mode="pytest"   -> python -m pytest (uses the template's pytest.ini)
    mode="selftest" -> python run.py --selftest (no pytest needed)
    """
    python = python or sys.executable
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{t.name}.log"
    junit = out_dir / f"{t.name}.junit.xml"
    if junit.exists():
        junit.unlink()

    if mode == "selftest":
        if not t.has_run_py:
            return RunResult(t.tid, t.name, mode, "NO_TESTS", message="run.py not found")
        cmd = [python, "run.py", "--selftest"]
    else:
        cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
        cmd += extra_args or []

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=t.path, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        log_file.write_text(f"$ {' '.join(cmd)}\nTIMEOUT after {timeout}s\n{e.stdout or ''}", encoding="utf-8")
        return RunResult(t.tid, t.name, mode, "TIMEOUT", seconds=round(time.perf_counter() - start, 2),
                         log_file=str(log_file), message=f"timed out after {timeout}s")
    secs = round(time.perf_counter() - start, 2)
    log_file.write_text(f"$ {' '.join(cmd)}\n(cwd={t.path})\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
                        encoding="utf-8")

    res = RunResult(t.tid, t.name, mode, "PASS", seconds=secs, exit_code=proc.returncode, log_file=str(log_file))
    if mode == "selftest":
        res.status = "PASS" if proc.returncode == 0 else "FAIL"
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("PASS", "FAIL"))]
        res.passed = sum(ln.startswith("PASS") for ln in lines)
        res.failed = sum(ln.startswith("FAIL") for ln in lines)
        if proc.returncode != 0 and not res.failed:
            res.status, res.message = "ERROR", _last_line(proc.stderr)
        return res

    if not junit.exists():
        res.status, res.message = "ERROR", _last_line(proc.stderr or proc.stdout) or "pytest produced no report"
        return res
    c = parse_junit(junit)
    res.failed, res.errors, res.skipped, res.xfailed = c["failed"], c["errors"], c["skipped"], c["xfailed"]
    res.passed = c["tests"] - c["failed"] - c["errors"] - c["skipped"] - c["xfailed"]
    if c["tests"] == 0:
        res.status = "NO_TESTS" if proc.returncode == 5 else "ERROR"
        res.message = _last_line(proc.stdout) or _last_line(proc.stderr)
    elif c["failed"]:
        res.status = "FAIL"
    elif c["errors"] or proc.returncode not in (0,):
        res.status = "ERROR"
        res.message = _last_line(proc.stdout)
    return res


def _last_line(text: str) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""
