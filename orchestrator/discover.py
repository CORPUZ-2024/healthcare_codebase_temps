"""
discover.py — find templates by FOLDER CONVENTION only.

A template is any folder directly under ``templates/`` that:
  * matches the name pattern ``t<2 digits>_<name>`` (e.g. ``t05_tcoc_pmpm_mlr``)
  * contains a ``tests/`` folder

Folders starting with ``_`` (e.g. ``_skeleton``) are ignored.

The orchestrator NEVER imports template code. ``template.yaml`` is read as plain
text with a tiny parser (no PyYAML needed) and only used for filtering
(``--group``, ``--intent``) and for display names.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TEMPLATE_DIR_RE = re.compile(r"^t\d{2}_[a-z0-9_]+$")


@dataclass
class TemplateInfo:
    """Everything the orchestrator knows about one template (all from disk)."""
    tid: str                 # "t05"
    name: str                # "t05_tcoc_pmpm_mlr"
    path: Path               # absolute folder path
    meta: dict = field(default_factory=dict)   # parsed template.yaml (may be empty)
    has_run_py: bool = False

    @property
    def group(self) -> str:
        return str(self.meta.get("type", "")).upper()

    @property
    def intents(self) -> list[str]:
        v = self.meta.get("intents", [])
        return [s.upper() for s in (v if isinstance(v, list) else [v])]

    @property
    def title(self) -> str:
        return str(self.meta.get("title", self.name))


def parse_simple_yaml(text: str) -> dict:
    """Parse the flat subset of YAML used by template.yaml.

    Supported lines:
        key: value
        key: [a, b, c]
        # comments and blank lines
    Anything nested is ignored. This keeps the orchestrator stdlib-only.

    >>> parse_simple_yaml("id: t05\\nintents: [MEASURE, VALUE]\\n")
    {'id': 't05', 'intents': ['MEASURE', 'VALUE']}
    """
    out: dict = {}
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip()
        if not line or line.lstrip().startswith("#") or line.startswith((" ", "\t", "-")):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            out[key.strip()] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        else:
            out[key.strip()] = val.strip("'\"")
    return out


def discover(templates_root: Path) -> list[TemplateInfo]:
    """Return templates found under ``templates_root``, sorted by id."""
    templates_root = Path(templates_root)
    if not templates_root.is_dir():
        raise FileNotFoundError(f"templates folder not found: {templates_root}")
    found: list[TemplateInfo] = []
    for d in sorted(templates_root.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        if not TEMPLATE_DIR_RE.match(d.name) or not (d / "tests").is_dir():
            continue
        meta_file = d / "template.yaml"
        meta = parse_simple_yaml(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        found.append(TemplateInfo(tid=d.name[:3], name=d.name, path=d.resolve(), meta=meta,
                                  has_run_py=(d / "run.py").exists()))
    return found


def apply_filters(templates: list[TemplateInfo], only: str | None = None,
                  group: str | None = None, intent: str | None = None) -> list[TemplateInfo]:
    """Filter by comma-separated ids (``t03,t05``), type letter (``D``), or intent (``VALUE``)."""
    out = templates
    if only:
        wanted = {w.strip().lower()[:3] for w in only.split(",") if w.strip()}
        out = [t for t in out if t.tid in wanted]
    if group:
        out = [t for t in out if t.group == group.upper()]
    if intent:
        out = [t for t in out if intent.upper() in t.intents]
    return out
