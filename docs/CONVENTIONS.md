# Conventions

These rules apply to every template in `templates/`. They exist so you can open any
template cold and know where everything is.

## 1. Folder layout (identical in every template)

```
tNN_<name>/
├── README.md          what / why / when, how to run, standard vs. alternative method,
│                      test-data sources, caveats, "explain it to Finance"
├── template.yaml      metadata only (id, type, intents, volume, stacks, settings)
├── requirements.txt   only the libraries THIS template needs
├── pytest.ini         makes `pytest` inside the folder self-contained
├── run.py             python run.py            -> full demo, writes outputs/
│                      python run.py --selftest -> plain-assert checks, no pytest needed
├── <name>/            the package (name = folder minus the tNN_ prefix, unique repo-wide)
│   ├── config.py      every tunable parameter in one dataclass
│   ├── data.py        seeded synthetic generator + loader for the public test file
│   ├── checks.py      data-quality and assumption checks -> list of Finding
│   └── methods.py     the analysis: STANDARD method + ALTERNATIVE method
├── sql/               SQL twin of the core logic (DuckDB dialect) where it applies
├── tests/             pytest suite (conftest.py + test_tNN_*.py)
├── data/README.md     where to download the public test dataset; files here are gitignored
└── outputs/           written by run.py; gitignored
```

## 2. Standard + alternative methods

Every analytical step ships two implementations:

| | Standard | Alternative |
|---|---|---|
| What | the industry-default library/function (what a reviewer expects to see) | the second-most-useful approach |
| When | default | when a named caveat of the standard method bites |
| Where documented | docstring + README "Method choices" table | same, with the trade-off stated |

Tests compare the two on the same data, so you can see where they agree and where they don't.

## 3. Docstrings (NumPy style + beginner sections)

```
Summary line.

Healthcare context   why an analyst does this; who uses the output
Parameters / Returns column names, types, units
Steps                numbered; mirrors the code line by line
Example              small doctest where practical
Common mistakes      2-4 bullets
```

## 4. Column naming

| Suffix | Meaning | Example |
|---|---|---|
| `_id` | identifier (string, never numeric) | `member_id`, `npi_id` |
| `_dt` | date | `svc_from_dt` |
| `_cd` | code | `dx1_cd`, `hcpcs_cd`, `pos_cd` |
| `_amt` | money (USD) | `paid_amt` |
| `_cnt` | count | `visit_cnt` |
| `_flag` | 0/1 or bool | `readmit_30d_flag` |
| `_pct` / `_rate` | proportion 0-1 | `pdc_pct` |

## 5. Data policy

* Only generated data is committed. Public-use files are documented in each `data/README.md`
  and downloaded by you; they are gitignored.
* Any table that is published by CMS or licensed (HCC coefficients, NCQA value sets, etc.)
  ships as a clearly labelled `FAKE_` toy table with a pointer to the real source.
* Synthetic data is seeded, so every run and test is reproducible.

## 6. Independence

* A template never imports another template or the orchestrator.
* Shared helpers are **copied**, not imported. `orchestrator/tests/test_orch_independence.py`
  enforces this.
