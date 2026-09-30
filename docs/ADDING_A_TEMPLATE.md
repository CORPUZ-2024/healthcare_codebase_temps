# Adding a template

1. Copy `templates/_skeleton` to `templates/tNN_<your_name>` (next free number, lowercase, underscores).
2. Rename the package folder `skeleton/` to `<your_name>/` (must be unique across the repo) and
   update the imports in `run.py` and `tests/`.
3. Edit `template.yaml` (id, type A-G, intents, volume, stacks, settings).
4. Replace the example in `methods.py` with your STANDARD and ALTERNATIVE methods.
5. Write tests: at least one known-answer test you can check by hand, and one per check in `checks.py`.
6. Fill in `README.md`, including the public test-data source in `data/README.md`.
7. Run `python -m pytest` inside the folder, then `python orchestrator/run_all_tests.py --only tNN`.

The orchestrator finds the new template automatically; nothing needs registering.
