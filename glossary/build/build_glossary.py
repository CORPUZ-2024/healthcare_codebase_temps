"""Build glossary/niche_workflows_glossary.html from the entry list (E), full_entries.FULL and the gNN_*.py blocks.

    python glossary/build/build_glossary.py
"""
from pathlib import Path
import html, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_entries import FULL as _FULL  # noqa: E402

FACTORS = {
 "F1": ("Licensed", "Spec or code set needs a license (NCQA, PQA, AMA CPT, commercial groupers). Snippet shows an open approximation."),
 "F2": ("Restricted data", "Real data needs a DUA or registration (T-MSIS, RIF, ADI)."),
 "F3": ("Niche setting", "Narrow setting or infrequent for this role."),
 "F4": ("Heavy dependency", "Needs libraries beyond the shared floor."),
 "F5": ("Snippet-sized", "One function; a full pipeline would be over-engineering."),
 "F6": ("Sign-off domain", "Real use needs credentialed sign-off (actuarial, audit)."),
 "F7": ("Format specialist", "Interop format with no free real test data."),
}
UC = {
 "UC1": "Data standards & interoperability", "UC2": "Clinical classification & risk indices",
 "UC3": "Pharmacy", "UC4": "Advanced causal & HEOR", "UC5": "Actuarial & finance",
 "UC6": "Home-based care & Medicaid operations", "UC7": "AI/ML governance & privacy",
}
E = [
 ("G01","X12 835 remittance → paid/denied lines + CARC codes","UC1",["F7","F5"],"PAYER VBC HOSP","Parsing is segment-level string work with no free real 835 files to test against.","You start owning denial analytics or cash posting reconciliation."),
 ("G02","FHIR R4 bundle flattening (Synthea)","UC1",["F7","F4"],"VBC HOSP","Nested JSON resources need a resource-by-resource mapping; one flattener, not a pipeline.","An EHR/HIE feed becomes a primary data source."),
 ("G03","Fuzzy record linkage / member match","UC1",["F4"],"VBC HOME GOV","Needs a linkage library and a labelled truth set you won't have on synthetic data.","You merge caregiver-program rosters with payer eligibility regularly."),
 ("G04","Continuous enrollment with allowable gap","UC1",["F5"],"PAYER VBC","A single span-walking function reused by T04; kept here as a standalone reference.","—"),
 ("G05","Charlson / Elixhauser comorbidity index","UC2",["F5"],"VBC HOSP HEOR","Code-list lookup + weighted sum; one function.","Comorbidity adjustment becomes standard in your outcome studies."),
 ("G06","NYU ED algorithm (ED visit classification)","UC2",["F5","F1"],"PAYER VBC","Needs the published lookup file; the logic is a probability-weighted join.","ED avoidance becomes a contracted metric."),
 ("G07","AHRQ PQI potentially avoidable hospitalizations","UC2",["F5","F1"],"PAYER VBC GOV","AHRQ ships official software; the snippet is for intuition only.","A contract pays on PQI-based avoidable admissions."),
 ("G08","LACE readmission index","UC2",["F5"],"HOSP VBC HOME","Four-component additive score.","—"),
 ("G09","Provider attribution (plurality of E&M visits)","UC2",["F5"],"VBC","Contract-specific rules; the snippet shows the plurality pattern.","You must replicate a payer's attribution file."),
 ("G10","PDC adherence (Stars-style)","UC3",["F5","F1"],"PAYER RX","One well-defined calculation; the official measure specification is licensed by PQA.","Pharmacy adherence becomes a quality-gate metric in a contract."),
 ("G11","Opioid MME per day","UC3",["F5"],"PAYER RX","Needs the CDC conversion-factor file; logic is one join + arithmetic.","—"),
 ("G12","NDC normalization (10 → 11 digit) + package mapping","UC3",["F5"],"PAYER RX","Pure string normalization; also embedded in T00.","—"),
 ("G13","Synthetic control","UC4",["F4","F3"],"GOV HEOR","Needs an optimizer and many donor units; rare for a single program evaluation.","You evaluate a state- or county-level policy with few treated units."),
 ("G14","Regression discontinuity","UC4",["F3"],"HEOR","Requires a sharp eligibility cutoff in the data.","A program enrolls on a score threshold (e.g., acuity tier cutoff)."),
 ("G15","Instrumental variables (2SLS)","UC4",["F3"],"HEOR","Valid instruments are rare in operational data.","—"),
 ("G16","Competing risks (cumulative incidence)","UC4",["F4"],"HEOR HOME","Extension of T11 needing Aalen–Johansen / Fine–Gray tooling.","Death vs. hospitalization competes in your outcome studies."),
 ("G17","Multiple imputation (chained equations)","UC4",["F5"],"HEOR","Wraps one library call; the lesson is in the diagnostics.","—"),
 ("G18","Cost-effectiveness: ICER + Markov QALY","UC4",["F3"],"HEOR","Pharma/HTA-style evaluation; not typical for payer ROI work.","A payer or state asks for cost per QALY."),
 ("G19","Budget impact model","UC4",["F3"],"HEOR PAYER","Spreadsheet-style projection; a function, not a pipeline.","—"),
 ("G20","Two-part cost model (logit + Gamma GLM)","UC4",["F5"],"HEOR VBC","One modeling pattern; could fold into T06 later.","Zero-heavy cost outcomes appear in most ROI studies."),
 ("G21","Bootstrap CI for cost differences","UC4",["F5"],"VBC HEOR","One resampling function.","—"),
 ("G22","Credibility weighting (limited fluctuation)","UC5",["F6"],"PAYER","Actuarial domain; results need actuarial sign-off.","—"),
 ("G23","Stop-loss / risk-corridor pricing sketch","UC5",["F6"],"PAYER VBC","Pricing requires actuarial certification.","—"),
 ("G24","RADV-style sample & extrapolation","UC5",["F6","F2"],"PAYER","Audit methodology; real use needs medical-record review data.","—"),
 ("G25","EVV compliance + authorized-vs-delivered hours","UC6",["F3","F7"],"HOME GOV","State-specific rules and aggregator file formats vary by state.","Role confirms EVV data is in scope → promote into T15."),
 ("G26","Medicaid churn / redetermination spells","UC6",["F2","F3"],"PAYER GOV","Needs T-MSIS-like enrollment history to be realistic.","—"),
 ("G27","Dual-eligible identification (dual status codes)","UC6",["F2"],"PAYER GOV","Monthly dual status codes come from restricted files.","—"),
 ("G28","Population stability index (drift)","UC7",["F5"],"PAYER VBC HOME","One function comparing two distributions.","You deploy T03 scores to production."),
 ("G29","Subgroup calibration / fairness audit","UC7",["F5"],"PAYER VBC HOME","A diagnostic layer on top of T03 outputs.","—"),
 ("G30","Small-cell suppression (cell size < 11)","UC7",["F5"],"ALL","One function applied before any external release.","—"),
 ("G31","Clinical NLP with negation (rule-based)","UC7",["F4"],"VBC HOME HOSP","Needs NLP tooling and note text you won't have synthetically.","Care notes become an analysis source."),
]

# attach the source of each copy block (single source of truth: the gNN_*.py file that the tests execute)
FULL = {gid: {**meta, "code": open(Path(__file__).with_name(meta["file"]), encoding="utf-8").read()} for gid, meta in _FULL.items()}
missing = [e[0] for e in E if e[0] not in FULL]

def chip(f): return f'<span class="chip f" title="{html.escape(FACTORS[f][1])}">{f} · {FACTORS[f][0]}</span>'

def entry_html(e):
    gid,name,uc,fs,st,why,promote = e
    full = FULL.get(gid)
    head = (f'<header class="eh"><span class="gid">{gid}</span><h3>{html.escape(name)}</h3>'
            f'<div class="chips">{"".join(chip(f) for f in fs)}<span class="chip s">{st}</span></div></header>')
    meta = (f'<div class="meta"><div><span class="lbl">Why not a template</span>{html.escape(why)}</div>'
            f'<div><span class="lbl">Promote to template when</span>{html.escape(promote)}</div></div>')
    if not full:
        body = '<div class="pending">Copy block, caveats and common mistakes are written in build phase P6.</div>'
        cls = "stub"
    else:
        cls = "full"
        cav = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>" for a,b in full["caveats"])
        mis = "".join(f"<li>{html.escape(m)}</li>" for m in full["mistakes"])
        src = "".join(f'<li><a href="{u}" target="_blank" rel="noopener">{html.escape(t)}</a></li>' for t,u in full["sources"])
        body = (f'<div class="io">{full["io"]}</div>'
                f'<div class="codewrap"><div class="codebar"><span>Copy-paste block · Python · runs standalone, self-tests on <code>python file.py</code></span>'
                f'<button class="copy" type="button">Copy</button></div><pre><code>{html.escape(full["code"])}</code></pre></div>'
                f'<h4>Caveats + solutions</h4><table class="cav"><thead><tr><th>Caveat</th><th>Solution</th></tr></thead><tbody>{cav}</tbody></table>'
                f'<h4>Common mistakes</h4><ul class="mis">{mis}</ul>'
                f'<h4>Sources</h4><ul class="src">{src}</ul>')
    return (f'<article class="entry {cls}" id="{gid}" data-uc="{uc}" data-f="{" ".join(fs)}" data-s="{st}" '
            f'data-text="{html.escape((gid+" "+name+" "+why).lower())}">{head}{meta}{body}</article>')

sections = ""
toc = ""
for uc,label in UC.items():
    items = [e for e in E if e[2]==uc]
    sections += f'<section class="uc" data-uc="{uc}"><h2><span class="ucid">{uc}</span>{html.escape(label)} <span class="n">{len(items)}</span></h2>' + "".join(entry_html(e) for e in items) + "</section>"
    toc += f'<li class="tuc" data-uc="{uc}"><b>{html.escape(label)}</b><ul>' + "".join(
        f'<li data-id="{e[0]}"><a href="#{e[0]}">{e[0]} {html.escape(e[1])}</a>{" <i>●</i>" if e[0] in FULL else ""}</li>' for e in items) + "</ul></li>"

legend = "".join(f'<div class="lg"><span class="chip f">{k} · {v[0]}</span><span>{html.escape(v[1])}</span></div>' for k,v in FACTORS.items())
uc_opts = "".join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in UC.items())
f_opts = "".join(f'<option value="{k}">{k} · {v[0]}</option>' for k,v in FACTORS.items())
s_opts = "".join(f'<option value="{s}">{s}</option>' for s in ["PAYER","VBC","HOME","HOSP","RX","GOV","HEOR"])

tpl = open(Path(__file__).with_name("glossary_template.html"), encoding="utf-8").read()
out = (tpl.replace("{{SECTIONS}}", sections).replace("{{TOC}}", toc).replace("{{LEGEND}}", legend)
          .replace("{{UC_OPTS}}", uc_opts).replace("{{F_OPTS}}", f_opts).replace("{{S_OPTS}}", s_opts)
          .replace("{{N}}", str(len(E))).replace("{{NFULL}}", str(len(FULL))))
if not missing:   # all entries populated: the mock-up banner becomes a plain status line
    import re
    out = re.sub(r'<div class="banner">MOCK-UP.*?</div>',
                 f'<div class="banner">{len(FULL)} of {len(E)} entries populated. Every copy block self-tests with '
                 f'<code>python file.py</code> and is executed by glossary/test_glossary_blocks.py.</div>', out, flags=re.S)
open(Path(__file__).resolve().parents[1] / "niche_workflows_glossary.html", "w", encoding="utf-8").write(out)
print("ok", len(out), "| full entries:", len(FULL), "of", len(E), "| missing:", missing or "none")
