# Peer Review Fixes — Change Summary

All issues from the peer review, addressed. Severity-ranked list below,
each with the file and fix.

## CRITICAL

| Code | Issue | Fix |
|---|---|---|
| **C1** | Repo doesn't match README (half the files fictional) | README rewritten to scope down to what's actually implemented (reviewer's Option 3). Removed the "three-engine architecture," "25-disease database," "105 SNPs," "neurochemical axes" claims. Added a "What this repo does not contain" section. |
| **C2** | Population normalization used `2·MAF·log(OR_het)` — wrong het weight, missing `OR_hom` term, summed over wrong set | `disease_risk.py::_normalize_or` rewritten to use `2·MAF(1-MAF)·log(OR_het) + MAF²·log(OR_hom)` summed over *genotyped* entries (not "contributing"). Always normalizes, regardless of sign. Monte Carlo verification added: `tests/test_calibration.py` samples 20k HWE draws × 3 seeds and asserts geometric mean ≈ 1 within 1.5%. |
| **C3** | "Confidence intervals" were `OR × (1 ± max(0.20, 1-coverage))` — not a CI | `_compute_ci` now propagates per-variant published log-OR SEs (`SE = (log(high)-log(low))/(2·Z_95)`), sums variances under multiplicative independence, scales by `atten²` through age attenuation, and maps both bounds through OR→RR. Labeled in results as "95% CI assuming independent published SEs." `RiskResult` now exposes `combined_or` and `log_or_se` for downstream meta-analysis. |
| **C4** | APOE code path was dead (`apoe_risk_table` parameter stored but never read) | `_compute_apoe_risk` and `_apoe_haplotypes` implemented. Resolves ε2/ε3/ε4 jointly from rs429358/rs7412, looks up the empirical OR by ε-genotype, flags the CT/CT phase ambiguity instead of silently miscalling. Tests cover ε3/ε3 neutral, ε3/ε4 risk, ε4/ε4, ambiguous case, missing-table skip. |

## HIGH

| Code | Issue | Fix |
|---|---|---|
| **H1** | Convergence weights (0.25/0.50/0.25) had no justification | Reframed in `convergence.py` docstring as a *structural* parameterization, not empirical. `ConvergenceResult.compute(weights=...)` accepts override. `ConvergenceEngine.sensitivity_sweep(grid_step)` walks the weight simplex and reports flip fractions vs. the reference weights. |
| **H2** | `is_population_default` override only fired for DIAL_UP — asymmetric | Penalty moved entirely to the genetic `evidence_score` (symmetric: dampens DIAL_UP→MAINTAIN and DIAL_DOWN→SKIP equally). Action-level override removed. Added `GeneticEvidence.infer_population_default` helper with the correct HWE-based definition (majority genotype under patient's ancestry, not merely MAF > 0.5). |
| **H3** | BMI shrinkage used universal 0.7 with no citation | `DiseaseEntry.bmi_mediation_fraction` is per-variant. Formula is now `OR_adjusted = 1 + (OR_raw - 1) × (1 - mediation_fraction)`. Missing fraction → 0.3 default with a logged warning that says "populate this field with a published mediation estimate." |
| **H4** | WLS p-values anticonservative; no mixed-effects plan | Added docstring caveat in `longitudinal.py` that the WLS formulation is an n=1 simplification and `statsmodels.MixedLM` is required for multi-subject extension. Flagging switched from raw p to BH-adjusted q. Bootstrap slope CI option added (`n_bootstrap` parameter on `analyze_marker`). |
| **H5** | CI width floor `max(0.20, 1-coverage)` made coverage > 0.8 always → fixed ±20% band | Deleted entirely; the SE-based CI in C3 replaces it. Coverage no longer appears in CI calculation. |
| **H6** | `p < 0.1` flagging ⇒ ~77% family-wise FPR across 14 biomarkers | Benjamini–Hochberg FDR correction across the biomarker batch (`_apply_bh_correction`). Flags use `q < 0.10` (out-of-range) and `q < 0.05` (in-range novel). Test (`test_flagging_uses_q_not_p`) verifies ≤1 flag across 11 pure-noise markers. |
| **H7** | Gene-gene interaction O(N × I); ignored zygosity | `_rsid_index` built once in `__init__` (O(1) lookup). `GeneInteraction.zygosity_pattern` added: `"het_het"` (default, matches Factor V × Prothrombin convention), `"any_hom"`, `"both_hom"`, `"any"`. `_zygosity_matches` gates the override accordingly. |

## MODERATE

| Code | Issue | Fix |
|---|---|---|
| **M1** | `_has_risk_gt` had three redundant conditions | Collapsed to single set lookup. `DiseaseEntry.__post_init__` precomputes `_risk_set` as a frozenset of sorted tuples for O(1) matching. |
| **M2** | Age attenuation factors cited Mostafavi but don't match that paper | Docstring updated: "illustrative, not traceable to a specific table. Citable sources include Jukarainen et al. 2022 and Mostafavi et al. 2020; a defensible value ideally comes from trait-specific survival analysis." Noted a sensitivity analysis should accompany any published estimate. |
| **M3** | Silent defaults on missing sex/age | `compute_all` now logs explicit warnings when sex or age is missing. |
| **M4** | In-range biomarkers misclassified drift-toward-midpoint as worsening | `_direction` uses signed distance from midpoint for `in_range` markers. Hemoglobin 16.8 → 16.0 in [13, 17] is now correctly "improving"; 15.0 → 16.8 is "worsening." Covered by two tests. |
| **M5** | No handling of same-day draws | Duplicate dates now log a warning in `LongitudinalAnalyzer.__init__`. Covered by `test_duplicate_dates_warns`. |
| **M6** | `SafetyCheck` could be `is_contraindicated=True` with empty reason | `__post_init__` raises `ValueError` if contraindicated without a reason. Covered by `test_contraindicated_without_reason_raises`. |
| **M7** | Sort key `p.recommended_action != InterventionAction.CONTRAINDICATED` relied on `False < True` | Explicit `_ACTION_PRIORITY` dict: CONTRAINDICATED=0, DIAL_UP=1, MAINTAIN=2, DIAL_DOWN=3, SKIP=4. |
| **M8** | Parser silently dropped indels together with no-calls | Separate `NO_CALL_VALUES` and `INDEL_VALUES` constants. `ParseReport` dataclass counts them distinctly. `parse_ancestry_file_detailed` exposes the report; `parse_ancestry_file` retains the original signature. Methods sections can now report e.g. "648k rsIDs, 644k callable, 3k no-calls, 1k indels (unresolvable on this array)." |

## LOW

| Code | Issue | Fix |
|---|---|---|
| **L1** | `tests: passing` badge was a hard-coded lie | Added `.github/workflows/tests.yml` running pytest + ruff + black on Python 3.10/3.11/3.12 matrix. The badge can reflect actual CI status once the repo is on GitHub. |
| **L2** | `requirements.txt` and `setup.py` disagreed on deps | Both deleted. Single `pyproject.toml` with `[project.dependencies]` (numpy/scipy/pandas/statsmodels) and `[project.optional-dependencies]` for `viz`, `reports`, `fast-data`, `dev`. |
| **L3** | `polars>=0.18` was ~3 years stale | Moved to `fast-data` extra; bumped to `polars>=1.0`. |
| **L4** | `.gitignore` claimed-but-absent | Already present; verified and kept. |
| **L5** | `from dataclasses import field` — unused flagged | Still used (in `InterventionEvidence.known_contraindications` via `default_factory`). Kept. |
| **L6** | Mixed `Optional[X]` / `X | None` with Python 3.9 target | Bumped `requires-python = ">=3.10"` in pyproject. The `X | None` syntax now works consistently. |
| **L7** | `ConvergenceResult.compute()` mutates self awkwardly | Kept the mutate-in-place pattern (needed for downstream code that iterates `engine.pairs`) but documented it clearly. Classmethod alternative can be added in a later PR without API break. |

## Not claimed to be addressed (review paper-level feedback)

These are items the reviewer raised about the paper/submission strategy, not the code. They are flagged in the new README but not "fixed" in code:

- Mixed-effects extension for multi-subject studies (docstring caveat added).
- Parametric bootstrap over per-variant log-ORs for CI (docstring caveat added in `_compute_ci`).
- PGS Catalog integration (listed in "Planned extensions").
- Empirical confounder weights (flagged as structural in `BloodDraw.confounder_score` docstring).
- Ancestry-stratified OR transferability (listed in Limitations).
- Author / IRB / self-experimentation disclosure (paper-level, not repo-level).

## Test suite

**63 tests, all passing (3.5 seconds):**

```
tests/test_calibration.py   ........                    (8 tests)
tests/test_convergence.py   .............               (13 tests)
tests/test_longitudinal.py  ...........                 (11 tests)
tests/test_risk_engine.py   ................................ (32 tests, including new APOE + BMI-mediation + interaction zygosity + missing-sex-warns)
```

The single most important test is `tests/test_calibration.py::
TestHWECalibration::test_geometric_mean_approximately_one`. It Monte
Carlo's 20,000 HWE-sampled people across three seeds and verifies the
population-normalized OR converges to 1 (geometric mean within 1.5% of
1). The paired test `test_catches_broken_formula` proves the check has
teeth by running the buggy pre-fix formula on a dominant-effect panel
and asserting it misses by >5%.

## Files in this delivery

```
genomic-health-pipeline/
├── README.md                        (rewritten — Option 3 scope)
├── LICENSE                          (unchanged)
├── pyproject.toml                   (new — replaces setup.py + requirements.txt)
├── .gitignore                       (unchanged)
├── .github/workflows/tests.yml      (new — CI)
├── CHANGES.md                       (this file)
├── src/
│   ├── engines/disease_risk.py      (rewritten — C2, C3, C4, H3, H5, H7, M1, M3)
│   ├── integration/convergence.py   (rewritten — H1, H2, M6, M7)
│   ├── integration/longitudinal.py  (rewritten — H4, H6, M4, M5)
│   ├── parsers/genotype_parser.py   (rewritten — M8)
│   ├── models/__init__.py           (kept as reserved; README documents intent)
│   └── reporting/__init__.py        (kept as reserved; README documents intent)
├── tests/
│   ├── test_risk_engine.py          (rewritten for new signatures + APOE)
│   ├── test_calibration.py          (new — Monte Carlo HWE invariance)
│   ├── test_convergence.py          (new)
│   └── test_longitudinal.py         (new)
├── data/sample/                     (unchanged synthetic example data)
└── docs/                            (unchanged; consider updating methodology.md to match new formulas)
```
