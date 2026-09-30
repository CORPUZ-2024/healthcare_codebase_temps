# Orchestrator

Runs every template's test suite and writes a status report. It is an **observer**:

* discovers templates by folder convention (`templates/tNN_*/tests/`); `_skeleton` is ignored
* runs each template in its **own subprocess** with the template folder as working directory
* reads results only from the exit code and a JUnit XML file
* imports only the Python standard library — never template code

Delete the orchestrator and every template still runs. Delete a template and the
orchestrator simply finds one fewer.

## Usage

```bash
python orchestrator/run_all_tests.py                  # everything
python orchestrator/run_all_tests.py --list           # what was found
python orchestrator/run_all_tests.py --only t03,t05   # by id
python orchestrator/run_all_tests.py --group D        # by workflow type (A-G)
python orchestrator/run_all_tests.py --intent EXPLAIN # by intent
python orchestrator/run_all_tests.py --selftest       # no pytest? uses run.py --selftest
python orchestrator/run_all_tests.py --jobs 4         # parallel
python orchestrator/run_all_tests.py -k known_answer  # pass a -k filter to every suite
```

Windows shortcut: `run_all_tests.bat` (uses `.venv` if present).

## Output — `orchestrator/run_status/`

| File | Content |
|---|---|
| `LATEST.md`, `latest.json`, `latest.html` | status of the most recent run (overwritten, committed) |
| `history/<timestamp>/summary.*` | every run's summary (gitignored) |
| `history/<timestamp>/logs/<template>.log` | full stdout/stderr per template |
| `history/<timestamp>/logs/<template>.junit.xml` | per-test results |
| `run_log.csv` | one appended row per template per run (gitignored) |

Status values: `PASS`, `FAIL` (a test failed), `ERROR` (collection/import error or crash),
`TIMEOUT`, `NO_TESTS`. Exit code is 0 only when every template is `PASS`.

## Its own tests

`python -m pytest orchestrator` — discovery, runner, report, CLI exit codes, and the
**independence contract** (no template references the orchestrator or another template;
the orchestrator is stdlib-only; every template has the required files).
