# Test data — t09 study design and power

No dataset is needed: power, MDE and randomization work from assumptions, and every analytic result
is checked by simulation inside the template.

Where the **assumptions** usually come from:
* **Baseline rates** (readmission, ED use): your own claims (t01), or public benchmarks such as CMS Care
  Compare / Hospital Readmissions Reduction Program files (provider-data.cms.gov).
* **Variability of cost** (CV): your claims (t05), or AHRQ MEPS for a population-level shape (see t06's data README).
* **ICC** for cluster designs: prior data from the same practices, or published ICC compilations for the
  outcome and setting. Always run a range.

Files in this folder are gitignored.
