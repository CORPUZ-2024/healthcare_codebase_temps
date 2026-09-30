@echo off
REM Run every template's tests. Extra arguments are passed through, e.g.:
REM   run_all_tests.bat --only t03,t05
REM   run_all_tests.bat --selftest
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe orchestrator\run_all_tests.py %*
) else (
  python orchestrator\run_all_tests.py %*
)
