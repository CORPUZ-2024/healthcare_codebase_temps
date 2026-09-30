# Debugging without AI help — a checklist

1. **Read the last line of the traceback first**, then the first line that points into *your* file.
2. **Print the shape.** Most pandas bugs are grain bugs:
   `print(df.shape, df.columns.tolist()); print(df.head())` before and after every merge.
3. **Check the join didn't fan out.** `assert len(merged) == len(left)` after a left join on a key
   that should be unique on the right. If it fails: `right[key].duplicated().sum()`.
4. **Run one test, stop at the failure, drop into the debugger:**
   `python -m pytest tests/test_t05_methods.py::test_pmpm_known_answer -x --pdb`
   Inside pdb: `p df.head()`, `pp locals().keys()`, `u`/`d` to move frames, `c` to continue.
5. **Add a breakpoint in code:** put `breakpoint()` on the line before the problem, run
   `python run.py`, inspect variables, `n` (next line), `s` (step in), `q` (quit).
6. **Shrink the input.** Every generator takes `n=`; re-run with `n=20` until you can check by hand.
7. **Compare with the alternative method.** Each template has one; if they disagree a lot, the
   README's "Method choices" table tells you which assumption to check.
8. **Compare with the SQL twin** (`sql/`), which often makes a grain mistake obvious.
9. **Run the self-test** (`python run.py --selftest`) when pytest isn't installed.
10. **Common error messages**
    * `KeyError: 'member_id'` → column renamed or missing; print `df.columns`.
    * `ValueError: You are trying to merge on object and int64 columns` → IDs must be strings.
    * `SettingWithCopyWarning` → add `.copy()` after filtering.
    * `PerfectSeparationError` / huge coefficients → a predictor perfectly predicts the outcome.
    * `LinAlgError: Singular matrix` → duplicated or constant columns; check `df.nunique()`.
