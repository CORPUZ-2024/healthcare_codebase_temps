#!/usr/bin/env bash
# Run every template's tests. Extra arguments are passed through.
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
exec "$PY" orchestrator/run_all_tests.py "$@"
