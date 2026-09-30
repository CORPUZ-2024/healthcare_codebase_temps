"""Discovery, runner and report tests using throw-away fake templates in tmp_path."""
import json
import sys
import textwrap
from pathlib import Path

import pytest

from discover import apply_filters, discover, parse_simple_yaml
from report import write_all, run_meta, totals
from runner import run_template


def make_template(root: Path, name: str, test_body: str, yaml: str = "", run_py: str | None = None):
    d = root / name
    (d / "tests").mkdir(parents=True)
    (d / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (d / "tests" / f"test_{name}.py").write_text(textwrap.dedent(test_body))
    if yaml:
        (d / "template.yaml").write_text(yaml)
    if run_py is not None:
        (d / "run.py").write_text(textwrap.dedent(run_py))
    return d


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "templates"
    make_template(root, "t01_good", "def test_ok():\n    assert 1 + 1 == 2\n",
                  "id: t01\ntype: B\nintents: [DESCRIBE, MONITOR]\n",
                  run_py="import sys\nprint('PASS  one')\nsys.exit(0)\n")
    make_template(root, "t02_bad", "def test_ok():\n    assert True\n\ndef test_bad():\n    assert 1 == 2\n",
                  "id: t02\ntype: D\nintents: [VALUE]\n",
                  run_py="import sys\nprint('FAIL  one')\nsys.exit(1)\n")
    make_template(root, "t03_xfail", "import pytest\n\n@pytest.mark.xfail(strict=False)\ndef test_x():\n    assert False\n\ndef test_ok():\n    pass\n")
    make_template(root, "_skeleton", "def test_ok():\n    pass\n")          # ignored: leading underscore
    (root / "t04_no_tests").mkdir()                                           # ignored: no tests/
    (root / "notes").mkdir()                                                  # ignored: bad name
    return root


def test_parse_simple_yaml():
    y = parse_simple_yaml("id: t05\n# comment\nintents: [MEASURE, VALUE]\ntitle: 'PMPM'\n  nested: ignored\n")
    assert y == {"id": "t05", "intents": ["MEASURE", "VALUE"], "title": "PMPM"}


def test_discover_uses_folder_convention(fake_root):
    names = [t.name for t in discover(fake_root)]
    assert names == ["t01_good", "t02_bad", "t03_xfail"]


def test_filters(fake_root):
    ts = discover(fake_root)
    assert [t.tid for t in apply_filters(ts, only="t02,t03")] == ["t02", "t03"]
    assert [t.tid for t in apply_filters(ts, group="d")] == ["t02"]
    assert [t.tid for t in apply_filters(ts, intent="monitor")] == ["t01"]


def test_runner_pass_fail_xfail(fake_root, tmp_path):
    ts = {t.tid: t for t in discover(fake_root)}
    good = run_template(ts["t01"], tmp_path / "logs")
    bad = run_template(ts["t02"], tmp_path / "logs")
    xf = run_template(ts["t03"], tmp_path / "logs")
    assert (good.status, good.passed) == ("PASS", 1)
    assert (bad.status, bad.passed, bad.failed) == ("FAIL", 1, 1)
    assert (xf.status, xf.passed, xf.xfailed) == ("PASS", 1, 1)
    assert Path(bad.log_file).exists()


def test_runner_selftest_mode(fake_root, tmp_path):
    ts = {t.tid: t for t in discover(fake_root)}
    assert run_template(ts["t01"], tmp_path, mode="selftest").status == "PASS"
    assert run_template(ts["t02"], tmp_path, mode="selftest").status == "FAIL"
    assert run_template(ts["t03"], tmp_path, mode="selftest").status == "NO_TESTS"


def test_timeout(tmp_path):
    root = tmp_path / "templates"
    make_template(root, "t09_slow", "import time\n\ndef test_slow():\n    time.sleep(5)\n")
    r = run_template(discover(root)[0], tmp_path / "logs", timeout=1)
    assert r.status == "TIMEOUT"


def test_report_files(fake_root, tmp_path):
    ts = discover(fake_root)
    results = [run_template(t, tmp_path / "logs") for t in ts]
    meta = run_meta("pytest", None)
    status = tmp_path / "run_status"
    write_all(results, status, status / "history" / meta["run_id"], meta)
    data = json.loads((status / "latest.json").read_text(encoding="utf-8"))
    assert data["totals"]["overall"] == "FAIL"
    assert (status / "LATEST.md").read_text(encoding="utf-8").count("| t0") == 3
    assert (status / "run_log.csv").read_text(encoding="utf-8").count("\n") == 4          # header + 3 rows
    assert totals(results)["passed_templates"] == 2


def test_cli_exit_codes(fake_root, tmp_path):
    import run_all_tests
    status = str(tmp_path / "rs")
    assert run_all_tests.main(["--templates-root", str(fake_root), "--status-root", status, "--only", "t01"]) == 0
    assert run_all_tests.main(["--templates-root", str(fake_root), "--status-root", status]) == 1
    assert run_all_tests.main(["--templates-root", str(tmp_path / "missing"), "--status-root", status]) == 2
