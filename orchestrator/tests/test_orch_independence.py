"""
Independence contract, checked against the REAL repository.

C3  no template file mentions/imports the orchestrator
C4  no template imports another template's package
C5  the orchestrator imports only the standard library + its own modules
"""
import ast
import re
import sys
from pathlib import Path

ORCH = Path(__file__).resolve().parents[1]
REPO = ORCH.parent

TEMPLATES = REPO / "templates"
OWN = {"discover", "runner", "report", "run_all_tests", "conftest"}


def imported_modules(py: Path) -> set[str]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def template_dirs():
    return [d for d in TEMPLATES.iterdir() if d.is_dir() and re.match(r"^t\d{2}_", d.name)] if TEMPLATES.exists() else []


def test_templates_do_not_reference_orchestrator():
    offenders = []
    for d in template_dirs():
        for py in d.rglob("*.py"):
            if "orchestrator" in imported_modules(py) or "run_all_tests" in py.read_text(encoding="utf-8"):
                offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"templates must not reference the orchestrator: {offenders}"


def test_templates_do_not_import_each_other():
    dirs = template_dirs()
    # each template's own package = folder name without the tNN_ prefix
    packages = {d.name: d.name[4:] for d in dirs}
    offenders = []
    for d in dirs:
        others = {p for n, p in packages.items() if n != d.name} | {n for n in packages if n != d.name}
        for py in d.rglob("*.py"):
            bad = imported_modules(py) & others
            if bad:
                offenders.append(f"{py.relative_to(REPO)} -> {sorted(bad)}")
    assert not offenders, f"templates must be standalone: {offenders}"


def test_orchestrator_is_stdlib_only():
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for py in ORCH.glob("*.py"):
        extra = imported_modules(py) - stdlib - OWN
        assert not extra, f"{py.name} imports non-stdlib modules: {extra}"


def test_every_template_has_required_files():
    required = ["README.md", "template.yaml", "requirements.txt", "run.py", "pytest.ini", "tests"]
    missing = [f"{d.name}/{f}" for d in template_dirs() for f in required if not (d / f).exists()]
    assert not missing, f"missing required files: {missing}"
