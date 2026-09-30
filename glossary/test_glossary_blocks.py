"""
Executes every glossary copy block (glossary/build/gNN_*.py) exactly as a reader would - `python file.py`
in a clean working directory - and checks the glossary build is consistent.

    python -m pytest glossary                         # from the repo root
    python orchestrator/run_all_tests.py --glossary   # opt-in from the orchestrator
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parent / "build"
BLOCKS = sorted(BUILD.glob("g[0-9][0-9]_*.py"))
sys.path.insert(0, str(BUILD))


def _entry_ids() -> list[str]:
    """Glossary ids from the E list in build_glossary.py, read without executing the build."""
    tree = ast.parse((BUILD / "build_glossary.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "E":
            return [elt.elts[0].value for elt in node.value.elts]
    raise AssertionError("E list not found in build_glossary.py")


@pytest.mark.parametrize("block", BLOCKS, ids=[b.stem for b in BLOCKS])
def test_block_self_test_passes(block, tmp_path):
    proc = subprocess.run([sys.executable, str(block)], cwd=tmp_path, capture_output=True, text=True,
                          timeout=600, encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    assert "PASS" in proc.stdout and "FAIL" not in proc.stdout, out[-3000:]


def test_block_is_self_contained():
    """A copy block must not import other blocks, templates or the orchestrator (readers copy ONE file)."""
    for b in BLOCKS:
        tree = ast.parse(b.read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not any(m.startswith(("g0", "g1", "g2", "g3", "orchestrator", "full_entries")) for m in mods), (b.name, mods)


def test_every_entry_has_a_full_record_and_block():
    from full_entries import FULL
    ids = _entry_ids()
    assert len(ids) == 31 and set(ids) == set(FULL)
    for gid, meta in FULL.items():
        assert (BUILD / meta["file"]).exists(), gid
        assert meta["file"].startswith(gid.lower()), (gid, meta["file"])
        assert len(meta["caveats"]) >= 3 and len(meta["mistakes"]) >= 2 and meta["sources"], gid
        assert all(u.startswith(("https://", "../")) for _, u in meta["sources"]), gid


def test_built_page_is_complete():
    html = (Path(__file__).resolve().parent / "niche_workflows_glossary.html").read_text(encoding="utf-8")
    assert html.count('class="entry full"') == 31 and 'class="entry stub"' not in html
    assert "MOCK-UP" not in html
