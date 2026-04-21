# Genomic Health Analytics Pipeline

**An open-source methods framework for population-normalized polygenic risk
estimation with genotype × phenotype convergence scoring and
confounder-weighted longitudinal biomarker analysis.**



---

## Scope

This repository implements three methods modules. It is a **methods
framework**, not a clinical product:

1. **Polygenic disease risk** (`src/engines/disease_risk.py`) — Hardy-Weinberg
   population-normalized combined OR with SE-propagated 95% CIs, age/BMI/
   height/sex adjustment, OR→RR conversion, APOE joint-genotype handling,
   and zygosity-aware gene-gene interaction detection.
2. **Genotype × phenotype convergence scoring**
   (`src/integration/convergence.py`) — weighted combination of genetic,
   phenotypic, and intervention evidence with a safety override; produces
   DIAL UP / MAINTAIN / DIAL DOWN / SKIP / CONTRAINDICATED labels and a
   weight-sensitivity sweep across the score-weight simplex.
3. **Longitudinal biomarker trend analysis**
   (`src/integration/longitudinal.py`) — weighted least-squares regression
   on serial blood panels with confounder-inverse weighting, BH-FDR
   correction across the biomarker batch, and optional bootstrap slope CIs.

Supporting modules: `src/parsers/genotype_parser.py` (AncestryDNA/23andMe
raw genotype files, with separate no-call and indel accounting).

### What this repo does *not* contain

Earlier drafts of this README described a broader "three-engine"
architecture with a 25-disease database, a 105-SNP health-interpretation
engine, a behavioral-tendencies engine, rarity scanning, DOCX reports, and
Jupyter notebooks. None of those exist in the code today. The README has
been rescoped to what is actually implemented. Those components, if
implemented, will be added incrementally in their own PRs.

The database of disease entries is not committed. A `DiseaseEntry` has a
clearly-defined schema (see `src/engines/disease_risk.py`); populating it
from the PGS Catalog (Lambert et al. 2021) or an equivalent curated
source is the natural next step.

---

## Method summary

### Disease risk engine

The full pipeline (see module docstring for a step-by-step):

1. Parse user genotypes.
2. Match against the `DiseaseEntry` database.
3. Compute per-variant OR by zygosity (hom-nonrisk → 1.0, het → `or_het`,
   hom-risk → `or_hom`).
4. Apply variant-specific BMI-mediation shrinkage when `bmi_mediated=True`
   and a `bmi_mediation_fraction` is provided (defaults to 0.3 with a
   logged warning when missing — this is a placeholder, not a published
   value).
5. Combine per-variant ORs multiplicatively (gene-gene interactions can
   override with an empirically-measured combined OR; zygosity pattern
   is declared per-interaction and defaults to "het_het").
6. **Hardy-Weinberg population normalization**:

   ```
   E[log OR_i] = 2·MAF·(1-MAF)·log(OR_het) + MAF²·log(OR_hom)
   log(OR_normalized) = log(OR_combined) - Σ_i E[log OR_i]
   ```

   The sum is taken over every variant the user was *genotyped* at
   (including hom-nonrisk carriers, whose log-OR is zero but whose
   expectation must still be subtracted for the HWE invariant to hold —
   see `tests/test_calibration.py` for the Monte Carlo verification).
7. Base rate adjusted by age-decade multiplier, BMI category, and
   disease-category-specific height modifier.
8. Age attenuation of the genetic log-OR (illustrative stepwise factors;
   ideally trait-specific survival analysis per Jukarainen et al. 2022).
9. OR → RR: `RR = OR / (1 - P₀ + P₀·OR)`.
10. Absolute risk capped at 80%. 95% CI propagated from per-variant
    published log-OR SEs under multiplicative independence, scaled by the
    age-attenuation factor squared, mapped through the OR→RR transform at
    both bounds.

The CI is reported as "95% CI assuming independent published SEs" — it
does not capture between-study heterogeneity, and a parametric bootstrap
over per-variant log-ORs is the planned upgrade.

**APOE** is a locus-specific special case: ε2/ε3/ε4 diplotypes are
non-multiplicative and must be resolved from the joint rs429358/rs7412
genotype. The engine does this via a lookup table
(`apoe_risk_table`) keyed by ε-genotype; phase-ambiguous unphased calls
(CT/CT) are flagged and skipped rather than silently miscalled.

### Convergence scoring

The convergence score combines three 0–1 component scores (genetic,
phenotypic, intervention) with weights `(0.25, 0.50, 0.25)` by default.
The weighting is a *structural* parameterization of the inference
principle "posterior probability of active dysfunction = prior (genetic)
× likelihood update (phenotypic)" — it is not derived from data and
should not be presented as such. A `ConvergenceEngine.sensitivity_sweep`
method walks the weight simplex and reports the fraction of pairs whose
action label flips across the grid, so the fragility of the scoring can
be reported alongside the point estimate.

Safety is an absolute override (contraindications short-circuit the
score). Population-default variants are penalized *at the genetic score
level* — a symmetric penalty that applies equally to DIAL_UP and
DIAL_DOWN recommendations.

### Longitudinal analysis

WLS regression on serial draws with weights = `1 - confounder_score`
(confounder weights are a structural choice; they are not fit from data).
P-values are anticonservative at small n — flagging uses BH-adjusted
q-values across the full biomarker batch, and bootstrap slope CIs are
available as an alternative to t-based CIs. For multi-subject extensions
(n > 1 participant), `statsmodels.MixedLM` with subject as a grouping
variable is the correct tool — the WLS formulation here is an n=1
simplification.

In-range biomarkers (e.g., hemoglobin) use signed distance from the
range midpoint to determine direction, so a value drifting from 16.8 →
16.0 within a 13–17 range is correctly classified as *improving*, not
worsening.

---

## Installation

```bash
git clone https://github.com/pranithakondipamula/genomic-health-pipeline.git
cd genomic-health-pipeline
pip install -e ".[dev]"
pytest -v
```

Requires Python 3.10+.

---

## Project structure

```
genomic-health-pipeline/
├── README.md
├── LICENSE                           # MIT
├── pyproject.toml                    # Single source of truth for deps/metadata
├── .gitignore
├── src/
│   ├── parsers/
│   │   └── genotype_parser.py        # Raw genotype parser; separate no-call/indel accounting
│   ├── engines/
│   │   └── disease_risk.py           # Polygenic risk with HWE normalization + SE CIs
│   ├── integration/
│   │   ├── convergence.py            # Genotype × phenotype scoring with sensitivity sweep
│   │   └── longitudinal.py           # Confounder-weighted trends with BH-FDR
│   ├── models/                       # (reserved for future DiseaseEntry catalogs)
│   └── reporting/                    # (reserved for future report generators)
├── tests/
│   ├── test_risk_engine.py           # OR→RR, normalization, attenuation, interactions, APOE
│   ├── test_calibration.py           # Monte Carlo HWE-invariance verification
│   ├── test_convergence.py           # Weights, sensitivity, safety override
│   └── test_longitudinal.py          # BH-FDR, direction logic, bootstrap CIs
├── data/sample/                      # Synthetic example data (safe to commit)
└── docs/
    ├── methodology.md
    └── evidence_audit_framework.md
```

---

## Running the tests

```bash
pytest                      # all tests
pytest tests/test_calibration.py -v   # the HWE Monte Carlo check
```

The calibration test is the strongest single invariant check in the
repo. It samples 20,000 simulated people under HWE, runs their per-variant
ORs through the normalization, and verifies that the geometric mean of
normalized combined ORs converges to 1 (within 1.5%) across three
random seeds. It also runs a paired "negative control" test that
replaces the HWE formula with its common broken variant
(`2·MAF·log(OR_het)` — no `(1-MAF)` factor, no `OR_hom` term) on a
dominant-effect panel and verifies that the broken formula is caught.
This demonstrates the invariant check has teeth.

---

## Limitations and caveats

- **No published database.** The code accepts a `DiseaseEntry` list; no
  curated database is committed. PGS Catalog (Lambert et al. 2021) is the
  recommended source.
- **Weights are structural, not empirical.** Both the convergence weights
  (0.25 / 0.50 / 0.25) and the confounder weights (travel 0.4, pollution
  0.3, sleep 0.2, stress 0.1) are design choices. Sensitivity analysis is
  provided for the convergence weights; the confounder weights are a
  candidate for data-driven estimation in any multi-subject extension.
- **n=1 statistics are limited.** The WLS / t-based p-values are
  anticonservative at n=3–5 draws. BH-FDR correction across the batch
  mitigates this for flagging decisions; bootstrap CIs are preferred for
  reporting slopes.
- **OR transferability across ancestries is imperfect** (Martin et al.
  2019). Use ancestry-matched MAFs in the `DiseaseEntry` records; OR
  portability remains a known limitation of any GWAS-derived score.
- **BMI-mediation fractions are variant-specific and should be cited.**
  The default fallback of 0.3 is a placeholder, not a published value.
- **Not a medical device.** This is research/methods code. It does not
  provide diagnoses and must not replace professional medical advice.

---

## Planned extensions (not implemented)

- Parametric bootstrap over per-variant log-ORs for CI estimation.
- `statsmodels.MixedLM`-based multi-subject trend analysis (replacing the
  n=1 WLS fallback).
- PGS Catalog loader for the disease database.
- Clinical HTML / DOCX report generators (`src/reporting/`).
- Ancestry-stratified variant rarity module with gnomAD v4 subpopulation
  frequencies.

