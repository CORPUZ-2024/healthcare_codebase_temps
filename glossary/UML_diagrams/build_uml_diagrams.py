"""Build one draw.io UML diagram per template, plus a toolkit overview.

    python glossary/UML_diagrams/build_uml_diagrams.py

Structure (modules, public functions, Config fields, spec folders) is read from
each template's code with `ast`, so re-running after a code change keeps the
diagrams honest. Method roles and extension points come from extensions.py.
Standard library only, like the orchestrator.
"""
from __future__ import annotations

import ast
import html
import math
import re
from pathlib import Path
from xml.sax.saxutils import quoteattr

from extensions import EXTENSIONS, NEW_TEMPLATES, ROLES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEMPLATES = REPO / "templates"

# ---------------------------------------------------------------- styles
FILL = {"S": "#dae8fc", "A": "#ffe6cc", "N": "#f8cecc", "": "none"}
TAG = {"S": "[S] ", "A": "[A] ", "N": "[naive] ", "": ""}
EXT_FILL = {"model": "#d5e8d4", "alternative": "#fff2cc", "loader": "#e1d5e7",
            "check": "#f5f5f5", "spec": "#e6f2ff", "module": "#d5e8d4"}
SWIM = ("swimlane;fontStyle=0;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;"
        "startSize={start};horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;"
        "collapsible=0;marginBottom=0;html=1;whiteSpace=wrap;fillColor={fill};strokeColor={stroke};"
        "rounded=0;shadow=0;{extra}")
ROW = ("text;strokeColor=none;fillColor={fill};align=left;verticalAlign=top;spacingLeft=4;"
       "spacingRight=4;overflow=hidden;rotatable=0;points=[[0,0.5],[1,0.5]];portConstraint=eastwest;"
       "html=1;whiteSpace=wrap;fontSize=11;fontFamily=Consolas;{extra}")
SEP = ("line;strokeWidth=1;fillColor=none;align=left;verticalAlign=middle;spacingTop=-1;spacingLeft=3;"
       "spacingRight=3;rotatable=0;labelPosition=right;points=[];portConstraint=eastwest;")
ROW_H = 18
CHAR_W = 6.6  # Consolas 11px


class Diagram:
    def __init__(self, name: str):
        self.name = name
        self.cells: list[str] = []
        self.geo: dict[str, tuple] = {}
        self.n = 1

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def vertex(self, value, style, x, y, w, h, parent="1") -> str:
        cid = self._id()
        self.geo[cid] = (x, y, w, h)
        self.cells.append(
            f'<mxCell id="{cid}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" '
            f'parent="{parent}"><mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" '
            f'as="geometry"/></mxCell>')
        return cid

    def edge(self, src, dst, label="", style="", points=()) -> str:
        cid = self._id()
        base = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=open;endSize=10;"
                "dashed=1;fontSize=10;labelBackgroundColor=#ffffff;")
        self.cells.append(
            f'<mxCell id="{cid}" value={quoteattr(label)} style={quoteattr(base + style)} edge="1" '
            f'parent="1" source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry">'
            + ('<Array as="points">' + "".join(f'<mxPoint x="{x:.0f}" y="{y:.0f}"/>' for x, y in points)
               + '</Array>' if points else "")
            + '</mxGeometry></mxCell>')
        return cid

    def text(self, value, x, y, w, h, extra=""):
        return self.vertex(value, "text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;" + extra,
                           x, y, w, h)

    def uml_box(self, header, rows, x, y, w, fill="#ffffff", stroke="#000000", extra="",
                start=40, sections=None):
        """rows: list of (text, row_fill) ; sections: indices before which a separator goes."""
        sections = set(sections or [])
        heights = []
        for txt, _ in rows:
            plain = re.sub(r"<[^>]+>", "", txt)
            lines = max(1, math.ceil(len(html.unescape(plain)) * CHAR_W / (w - 12)))
            heights.append(lines * 14 + 4)
        total = start + sum(heights) + 8 * len(sections) + 4
        box = self.vertex(header, SWIM.format(start=start, fill=fill, stroke=stroke, extra=extra),
                          x, y, w, total)
        cy = start
        for i, ((txt, rf), h) in enumerate(zip(rows, heights)):
            if i in sections:
                self.vertex("", SEP, 0, cy, w, 8, parent=box)
                cy += 8
            self.vertex(txt, ROW.format(fill=FILL.get(rf, rf), extra=""), 0, cy, w, h, parent=box)
            cy += h
        return box, total

    def xml(self) -> str:
        body = "\n        ".join(self.cells)
        return f"""<mxfile host="drawio" type="device">
  <diagram name={quoteattr(self.name)} id="{re.sub(r'[^a-z0-9]', '', self.name.lower())[:20]}">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="2100" pageHeight="1600" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


# ---------------------------------------------------------------- code reading
def esc(s: str) -> str:
    return html.escape(s, quote=False)


def short_sig(fn: ast.FunctionDef, limit=58) -> str:
    args = [a.arg for a in fn.args.args]
    sig = f"{fn.name}({', '.join(args)})"
    return sig if len(sig) <= limit else sig[: limit - 2] + "…)"


def read_module(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    return funcs, classes


def class_fields(cls: ast.ClassDef):
    out = []
    for s in cls.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            ann = ast.unparse(s.annotation)
            val = ast.unparse(s.value) if s.value is not None else ""
            if len(val) > 22:
                val = val[:20] + "…"
            out.append(f"{s.target.id}: {ann}" + (f" = {val}" if val else ""))
    return out


def readme_rows():
    """ID -> (area, headline) from the top-level README template table."""
    rows = {}
    for line in (REPO / "README.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(t\d\d)\s*\|\s*\[([^\]]+)\][^|]*\|\s*([^|]+)\|\s*([^|]+)\|", line)
        if m:
            rows[m.group(1)] = (m.group(3).strip(), m.group(4).strip())
    return rows


# ---------------------------------------------------------------- per-template diagram
COL = {"new": (40, 330), "data": (400, 360), "methods": (790, 480), "checks": (1300, 380),
       "spec": (1710, 330)}
SKIP_DIRS = {"tests", "data", "outputs", "__pycache__", ".pytest_cache"}


def build_template(tdir: Path, meta) -> Diagram:
    tid, pkg_name = tdir.name.split("_", 1)
    pkg = tdir / pkg_name
    roles = ROLES.get(tid, {})
    role_of = {f: r for r, fs in roles.items() for f in fs}
    area, headline = meta.get(tid, ("", ""))
    d = Diagram(f"{tdir.name} UML")

    # title + legend
    d.text(f"<font style='font-size:22px'><b>{esc(tdir.name)}</b></font>"
           f"<br><font color='#555555'>{esc(area)} · standard vs. alternative: {esc(headline)}</font>"
           f"<br><font color='#777777' style='font-size:11px'>UML class/component view generated from the "
           f"code by glossary/UML_diagrams/build_uml_diagrams.py · green dashed lane = extension points</font>",
           20, 10, 1500, 70)
    legend = [("[S] standard method (README default)", "S"), ("[A] alternative method", "A"),
              ("[naive] shown only to expose bias", "N"), ("helper / shared step", "")]
    d.uml_box("<b>Legend</b>", legend, 1710, 10, 330, start=26)

    y0 = 110
    ids, bottoms = {}, {}

    # package frame (drawn first so it sits behind)
    frame = d.vertex(f"<b>package {esc(pkg_name)}</b>  (templates/{esc(tdir.name)})",
                     "shape=folder;fontStyle=0;spacingTop=8;tabWidth=330;tabHeight=24;tabPosition=left;"
                     "html=1;verticalAlign=top;align=left;spacingLeft=10;fillColor=#fafafa;"
                     "strokeColor=#999999;dashed=0;", 20, y0 - 10, 2040, 100)

    # column A: run.py, Config, small modules
    x, w = COL["new"]
    run_rows = [("python run.py → full demo, writes outputs/", ""),
                ("python run.py --selftest → plain asserts", "")]
    if (tdir / "run.py").exists():
        rfuncs, _ = read_module(tdir / "run.py")
        run_rows += [("+ " + esc(short_sig(f, 40)), "") for f in rfuncs]
    ids["run"], h = d.uml_box("«entrypoint»<br><b>run.py</b>", run_rows, x, y0 + 30, w,
                              fill="#f5f5f5", stroke="#666666")
    cy = y0 + 30 + h + 30
    _, ccls = read_module(pkg / "config.py")
    cfg = next(c for c in ccls if c.name == "Config")
    ids["config"], h = d.uml_box("«dataclass»<br><b>Config</b>  (config.py)",
                                 [(esc(f), "") for f in class_fields(cfg)], x, cy, w,
                                 fill="#fff9e6", stroke="#b8860b")
    cy += h + 30
    extra_mods = [p for p in sorted(pkg.glob("*.py"))
                  if p.name not in {"__init__.py", "config.py", "data.py", "methods.py", "checks.py"}]
    for p in extra_mods:
        funcs, _ = read_module(p)
        rows = [("+ " + esc(short_sig(f, 44)), "") for f in funcs] or [("(no public API)", "")]
        if p.stem == "sqltwin":
            rows.append(("→ executes sql/*.sql on DuckDB; tests assert parity with methods.py", ""))
        mid, h = d.uml_box(f"«module»<br><b>{esc(p.name)}</b>", rows, x, cy, w)
        ids[p.stem] = mid
        cy += h + 20
    bottoms["new"] = cy

    # column B: data.py
    x, w = COL["data"]
    funcs, _ = read_module(pkg / "data.py") if (pkg / "data.py").exists() else ([], [])
    gen =[f for f in funcs if f.name.startswith(("generate", "sample", "lag_pattern", "luhn",
                                                  "to_delta", "cross_section", "practice_panel",
                                                  "its_series", "immortal", "truncated", "raw_mean"))]
    load = [f for f in funcs if f not in gen]
    rows = [("+ " + esc(short_sig(f, 52)), "") for f in gen]
    rows += [("+ " + esc(short_sig(f, 52)), "") for f in load]
    ids["data"], h = d.uml_box("«module»<br><b>data.py</b>  seeded synthetic + public loaders",
                               rows or [("(no data.py — inputs are Config parameters)", "")], x, y0 + 30, w,
                               sections=[len(gen)] if gen and load else None)
    bottoms["data"] = y0 + 30 + h + 20

    # column C: methods.py
    x, w = COL["methods"]
    funcs, _ = read_module(pkg / "methods.py")
    order = {"S": 0, "A": 1, "N": 2, "": 3}
    funcs = sorted(funcs, key=lambda f: order[role_of.get(f.name, "")])
    rows, breaks, prev = [], [], None
    for f in funcs:
        r = role_of.get(f.name, "")
        if prev is not None and r != prev:
            breaks.append(len(rows))
        rows.append((TAG[r] + "+ " + esc(short_sig(f, 62)), r))
        prev = r
    ids["methods"], h = d.uml_box("«module»<br><b>methods.py</b>  the analysis", rows, x, y0 + 30, w,
                                  sections=breaks)
    bottoms["methods"] = y0 + 30 + h + 20

    # column D: Finding + checks.py
    x, w = COL["checks"]
    funcs, classes = read_module(pkg / "checks.py")
    finding = next((c for c in classes if c.name == "Finding"), None)
    cy = y0 + 30
    if finding:
        ids["finding"], h = d.uml_box("«dataclass»<br><b>Finding</b>",
                                      [(esc(f), "") for f in class_fields(finding)], x, cy, w,
                                      fill="#fff9e6", stroke="#b8860b")
        cy += h + 40
    ids["checks"], h = d.uml_box("«module»<br><b>checks.py</b>  data-quality & assumption checks",
                                 [("+ " + esc(short_sig(f, 52)), "") for f in funcs], x, cy, w)
    bottoms["checks"] = cy + h + 20

    # column E: spec / sql folders
    x, w = COL["spec"]
    cy = y0 + 30
    spec_ids = []
    for sub in sorted(p for p in tdir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS
                      and p.name != pkg_name and not p.name.startswith(".")):
        files = sorted(f.name for f in sub.iterdir() if f.is_file() and not f.name.startswith("."))
        if not files:
            continue
        kind = "SQL twin (DuckDB)" if sub.name == "sql" else "spec / reference files"
        sid, h = d.uml_box(f"«folder»<br><b>{esc(sub.name)}/</b>  {kind}",
                           [(esc(f), "") for f in files], x, cy, w, fill="#eef5ff", stroke="#6c8ebf")
        ids["dir_" + sub.name] = sid
        spec_ids.append(sid)
        cy += h + 20
    loose = sorted(f.name for f in tdir.glob("*.y*ml") if f.name != "template.yaml")
    loose += sorted(f.name for f in tdir.glob("*.md") if f.name != "README.md")
    if loose:
        sid, h = d.uml_box("«files»<br><b>template root</b>  specs", [(esc(f), "") for f in loose],
                           x, cy, w, fill="#eef5ff", stroke="#6c8ebf")
        spec_ids.append(sid)
        ids["dir_root"] = sid
        cy += h + 20
    oid, h = d.uml_box("«artifact»<br><b>outputs/</b>", [("CSV / PNG written by run.py (gitignored)", "")],
                       x, cy, w, fill="#f5f5f5", stroke="#666666", start=40)
    ids["outputs"] = oid
    cy += h + 20
    bottoms["spec"] = cy

    # internal dependencies (two clear tracks above the boxes: y0+4 and y0+18)
    rx, ry, rw, rh = d.geo[ids["run"]]
    mx, my, mw, mh = d.geo[ids["methods"]]
    dx, dy, dw, dh = d.geo[ids["data"]]
    d.edge(ids["run"], ids["config"], "«reads»")
    d.edge(ids["run"], ids["data"], "«generate / load»", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.edge(ids["run"], ids["methods"], "«calls»", "exitX=0.75;exitY=0;entryX=0.25;entryY=0;",
           points=[(rx + rw * 0.75, y0 + 18), (mx + mw * 0.25, y0 + 18)])
    d.edge(ids["data"], ids["methods"], "DataFrames", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
           "endArrow=block;endFill=1;dashed=0;")
    if "finding" in ids:
        d.edge(ids["checks"], ids["finding"], "«creates» list[Finding]", "exitX=0.5;exitY=0;entryX=0.5;"
               "entryY=1;endArrow=diamond;endFill=0;")
    d.edge(ids["methods"], ids["checks"], "results checked", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    for k in ("dir_measures", "dir_reference", "dir_contracts", "dir_instruments", "dir_root",
              "dir_models"):
        if k in ids:
            sx, sy, sw, sh = d.geo[ids[k]]
            d.edge(ids["data"], ids[k], "«loads»", "exitX=0.75;exitY=0;entryX=0;entryY=0.15;",
                   points=[(dx + dw * 0.75, y0 + 4), (sx - 20, y0 + 4), (sx - 20, sy + sh * 0.15)])
            break

    # extension lanes
    content_bottom = max(bottoms.values())
    lane_y = content_bottom + 70
    lanes = {}
    for ext in EXTENSIONS.get(tid, []):
        lanes.setdefault(ext[0], []).append(ext)
    lane_target = {"new": "run", "data": "data", "methods": "methods", "checks": "checks",
                   "spec": spec_ids[0] if spec_ids else "outputs"}
    lane_title = {"new": "new module in package", "data": "extends data.py (new loaders)",
                  "methods": "extends methods.py (new models)", "checks": "extends checks.py",
                  "spec": "new spec files (no code)"}
    lane_max = lane_y
    for lane, exts in lanes.items():
        x, w = COL[lane]
        cy = lane_y + 40
        boxes = []
        for (_, stereo, name, sig, use, pairs) in exts:
            rows = [("+ " + esc(sig), "")]
            rows.append(("<i>Use:</i> " + esc(use), ""))
            if pairs:
                rows.append(("<i>Pairs with:</i> " + esc(pairs), ""))
            bid, h = d.uml_box(f"«{stereo}»<br><b>{esc(name)}</b>", rows, x + 10, cy, w - 20,
                               fill=EXT_FILL[stereo], stroke="#82b366", extra="dashed=1;")
            boxes.append(bid)
            cy += h + 14
        lane_h = cy - lane_y + 6
        lane_id = d.vertex(f"<b>EXTENSION POINT</b> · {esc(lane_title[lane])}",
                           "rounded=1;arcSize=2;dashed=1;dashPattern=8 4;strokeColor=#82b366;strokeWidth=2;"
                           "fillColor=#f3faf0;html=1;verticalAlign=top;align=left;spacingLeft=10;"
                           "spacingTop=6;fontColor=#2d6a2d;",
                           x, lane_y, w, lane_h)
        # lane must sit behind its boxes: move it before the first box cell
        lane_cell = d.cells.pop()
        first = next(i for i, c in enumerate(d.cells) if f'id="{boxes[0]}"' in c)
        d.cells.insert(first, lane_cell)
        tgt = ids.get(lane_target[lane], lane_target[lane])
        if lane in ("data", "methods", "checks"):
            d.edge(lane_id, tgt, "«extends»", "exitX=0.5;exitY=0;entryX=0.5;entryY=1;strokeColor=#2d6a2d;"
                   "fontColor=#2d6a2d;endArrow=block;endFill=0;edgeStyle=none;strokeWidth=2;")
        elif lane == "spec":
            d.edge(lane_id, tgt, "«adds»", "exitX=0.5;exitY=0;entryX=1;entryY=0.5;strokeColor=#2d6a2d;"
                   "fontColor=#2d6a2d;endArrow=block;endFill=0;strokeWidth=2;")
        else:
            d.edge(lane_id, ids["run"], "«wired from run.py»", "exitX=0;exitY=0.1;entryX=0;entryY=0.5;"
                   "strokeColor=#2d6a2d;fontColor=#2d6a2d;endArrow=block;endFill=0;strokeWidth=2;")
        lane_max = max(lane_max, lane_y + lane_h)

    # how-to note at the bottom
    d.text("<b>How to add an extension</b>: put the new function in the lane's module with a "
           "STANDARD/ALTERNATIVE docstring, add its knobs to <code>Config</code>, a <code>check_*</code> "
           "returning <code>Finding</code> for its assumptions, and a test in <code>tests/</code> that compares "
           "it with the current method on the seeded data (docs/ADDING_A_TEMPLATE.md, docs/CONVENTIONS.md §2). "
           "Templates never import each other — copy helpers, as prep.py does from t00.",
           40, lane_max + 20, 1600, 50, "fontSize=12;fontColor=#444444;")

    # resize package frame to wrap the code columns
    d.cells[[i for i, c in enumerate(d.cells) if f'id="{frame}"' in c][0]] = d.cells[
        [i for i, c in enumerate(d.cells) if f'id="{frame}"' in c][0]].replace(
        'height="100"', f'height="{content_bottom - y0 + 20:.0f}"')
    return d


# ---------------------------------------------------------------- overview
GROUPS = [
    ("Data foundation", ["t00"]),
    ("Population health & risk", ["t01", "t02", "t03", "t04"]),
    ("Health economics & VBC", ["t05", "t06", "t07", "t08"]),
    ("Evidence generation", ["t09", "t10", "t11", "t12"]),
    ("Performance & operations", ["t13", "t14", "t15"]),
]
HANDOFFS = [("t00", "t01", "prep.py copied"), ("t00", "t05", "prep.py copied"),
            ("t02", "t05", "RAF for risk-adjusted PMPM"), ("t03", "t06", "who to enrol"),
            ("t04", "t07", "quality gate"), ("t05", "t07", "PMPM / benchmark"),
            ("t08", "t05", "completed claims"), ("t09", "t10", "design → analysis"),
            ("t10", "t06", "causal method"), ("t11", "t03", "time-to-event risk"),
            ("t13", "t04", "provider measures"), ("t14", "t09", "experiment readout")]


def build_overview(dirs, meta) -> Diagram:
    d = Diagram("Toolkit overview")
    d.text("<font style='font-size:22px'><b>healthcare_codebase_temps — toolkit overview</b></font><br>"
           "<font color='#555555'>16 standalone templates. Arrows are <i>conceptual hand-offs</i> "
           "(outputs of one feed the next, or helpers are copied) — templates never import each other. "
           "Dashed green boxes: candidate new templates that would extend the portfolio.</font>",
           20, 10, 1700, 60)
    names = {p.name.split("_", 1)[0]: p.name for p in dirs}
    ids, x = {}, 30
    for gname, tids in GROUPS:
        gh = 60 + len(tids) * 110
        d.vertex(f"<b>{esc(gname)}</b>", "rounded=1;arcSize=3;html=1;verticalAlign=top;fillColor=#fafafa;"
                 "strokeColor=#999999;fontSize=13;spacingTop=6;", x, 100, 320, gh)
        for i, t in enumerate(tids):
            area, head = meta.get(t, ("", ""))
            ids[t], _ = d.uml_box(f"<b>{esc(names[t])}</b>", [(esc(head), "")], x + 15, 145 + i * 110,
                                  290, fill="#dae8fc", stroke="#6c8ebf", start=26)
        x += 350
    for a, b, lbl in HANDOFFS:
        d.edge(ids[a], ids[b], lbl, "strokeColor=#888888;fontColor=#555555;")
    y = 700
    d.text("<b>EXTENSION POINT · new templates</b>", 30, y, 600, 24, "fontColor=#2d6a2d;fontSize=14;")
    for i, (name, use, pairs) in enumerate(NEW_TEMPLATES):
        rows = [("<i>Use:</i> " + esc(use), "")]
        if pairs:
            rows.append(("<i>Pairs with:</i> " + esc(pairs), ""))
        d.uml_box(f"«proposed template»<br><b>{esc(name)}</b>", rows, 30 + i * 345, y + 35, 325,
                  fill="#d5e8d4", stroke="#82b366", extra="dashed=1;")
    d.text("Each new template copies <code>templates/_skeleton</code> and follows docs/ADDING_A_TEMPLATE.md; "
           "the orchestrator discovers it by folder name with no registration.", 30, y + 200, 1500, 30,
           "fontSize=12;fontColor=#444444;")
    return d


def main():
    meta = readme_rows()
    dirs = sorted(p for p in TEMPLATES.iterdir() if p.is_dir() and re.match(r"t\d\d_", p.name))
    written = []
    out = HERE / "00_toolkit_overview.drawio"
    out.write_text(build_overview(dirs, meta).xml(), encoding="utf-8")
    written.append(out)
    for t in dirs:
        out = HERE / f"{t.name}.drawio"
        out.write_text(build_template(t, meta).xml(), encoding="utf-8")
        written.append(out)
    for p in written:
        print("wrote", p.relative_to(REPO))


if __name__ == "__main__":
    main()
