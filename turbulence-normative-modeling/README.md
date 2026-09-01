# Turbulent dynamics and normative modelling of depressive symptoms

Analysis code for a study of turbulent brain dynamics (TD) in resting-state fMRI,
using normative modelling to test whether covariate-corrected deviation scores
predict depressive symptoms better than the raw features.

The analysis has two phases. Phase 1 characterises how TD features vary with age,
sex, and head motion across spatial scales in a healthy control cohort. Phase 2
uses those results to specify a Bayesian normative model, and evaluates the
resulting deviation scores as predictors of depressive symptoms.

The preregistration is at [osf.io/yjc39](https://osf.io/yjc39/).

## Turbulent dynamics features

All three features are built on the local Kuramoto order parameter under an
exponential distance rule, `W_ij = exp(-λ · d_ij)`, evaluated at 11 spatial
scales from λ = 0.01 to λ = 0.31 (roughly 100 mm down to 3 mm).

| Feature | What it measures | Supplementary text |
|---|---|---|
| Amplitude turbulence | Variability of local synchronisation over space and time | S1.1 |
| Information cascade flow | Efficiency of coupling between adjacent spatial scales | S1.2 |
| Information transfer | Spatial decay rate of order-parameter correlations | S1.3 |

## Repository layout

```
src/
  config.py            study parameters, field identifiers, output paths
  parcellation.py      parcel centroids, distance matrix, network index, ROI QC
  turbulence.py        the three TD features
  io_timeseries.py     loading parcellated BOLD time series
  cohort.py            cohort derivation, exclusions, covariate tests, matching
  phase1_lmm.py        mixed-effects models of demographic and motion effects
  normative_model.py   Bayesian normative model and deviation scores
  prediction.py        classification, cross-validation, permutation testing
  plotting.py          figures

scripts/
  01_compute_td_features.py     compute TD features for every participant
  02_phase1_healthy_trends.py   Phase 1 mixed-effects analysis
  03_fit_normative_model.py     fit the normative model, derive deviation scores
  04_phase2_prediction.py       matched-control classification
```

Every study-level parameter lives in `src/config.py`: the λ range, the TR and
filter band, the RDS cut-offs, the sampler settings, and the classifier
hyperparameters. Nothing is hard-coded in the analysis scripts, so the parameters
can be checked against the Methods in one place.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Phase 1 fits mixed-effects models through `pymer4`, which calls `lme4` in R via
`rpy2`. This needs a working R installation with `lme4` and `lmerTest`:

```bash
conda install -c conda-forge r-base r-lme4 r-lmertest
conda install -c ejolly -c conda-forge pymer4
```

Phases 2 and 3 do not depend on R.

## Data

This code uses UK Biobank imaging and phenotype data, which cannot be
redistributed. Access is granted through the
[UK Biobank Access Management System](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access).
The scripts read from paths set in `src/config.py`, overridable by environment
variable:

```bash
export TD_DATA_DIR=/path/to/data
export TD_RESULTS_DIR=/path/to/results
export TD_FIGURES_DIR=/path/to/figures
```

Two input tables are expected in `TD_DATA_DIR`:

- `phenotypes.csv` — one row per participant with the UK Biobank fields listed in
  `config.Fields`.
- `icd10.csv` — ICD-10 diagnoses, used for the neurological (Chapter VI, G00-G99)
  and psychiatric (Chapter V, F00-F99) exclusions.

Parcellated time series are expected as one directory per participant containing
`fMRI.Schaefer7n1000p.csv.gz` and `fMRI.Tian_Subcortex_S4_3T.csv.gz`. If your
extraction of the imaging archives is laid out differently, adapt
`SubjectFileLocator.resolve` in `src/io_timeseries.py` or pass any callable that
maps a participant id to the two file paths.

The preprocessing that produces these time series (MELODIC ICA+FIX denoising,
parcellation) is described in Mansour L. et al. (2023) and is not repeated here.

## Running the analysis

```bash
# 1. Compute TD features. This is the expensive step; results are written in
#    chunks so an interrupted run can be resumed with --start-chunk.
python scripts/01_compute_td_features.py --timeseries-root /path/to/timeseries

# 2. Phase 1: healthy trends across age, sex, and head motion.
python scripts/02_phase1_healthy_trends.py

# 3. Phase 2a: choose a model form by ELPD, then fit it and derive deviations.
python scripts/03_fit_normative_model.py --compare
python scripts/03_fit_normative_model.py --parameterisation spline_motion_only

# 4. Phase 2b: matched-control classification of depressive symptoms.
python scripts/04_phase2_prediction.py
```

Each script takes `--help`. Useful flags: `--skip-networks` on step 1 computes
whole-brain features only and is much faster; `--permutations 0` on step 4 skips
the null distribution.

## How the code maps onto the Methods

**Participant selection.** `cohort.load_phenotypes` derives the RDS score, the
bipolar screen, antidepressant use, and the depression recurrence-severity
grouping. `apply_common_exclusions` then removes nervous system disease, a
positive bipolar screen, and the excluded scanning site, printing the count lost
at each step so the numbers can be checked against the participant flowchart.
`select_healthy_controls` applies the strict Phase 1 criteria and
`select_depressive_symptoms` applies the RDS > 8 plus professional-consultation
criterion for Phase 2.

**Covariate comparison.** `cohort.compare_covariates` runs Fisher's exact test
for sex, Welch's t-test for age, and a Mann-Whitney U test for head motion,
reporting Hedges' g or Cramer's V.

**Phase 1.** `phase1_lmm.fit_feature_lmm` fits, per feature,

```
feature ~ motion * C(scale) + age_centered * sex * C(scale) + (1 | subject)
```

Scale enters as a categorical factor so interactions can be read at each λ and
non-linear effects of scale are absorbed rather than assumed away. The omnibus
F-test is FDR-corrected and partial eta squared is reported, labelled against
Cohen's benchmarks (small 0.01, medium 0.06, large 0.14).

**Phase 2, normative model.** `normative_model.compare_parameterisations` fits a
linear model and two spline variants and ranks them by expected log predictive
density from leave-one-out cross-validation. `fit_normative_model` then fits the
winner on the healthy training sample using the No-U-Turn Sampler through bambi,
with four chains of 2,000 draws after 1,000 tuning draws and bambi's default
weakly informative priors. `deviation_scores` applies the fitted model to a new
cohort and returns, per participant, the z-score of the observed feature against
the predicted mean and residual standard deviation.

**Phase 2, prediction.** `cohort.match_controls` draws controls matching the
cases' joint covariate distribution; head motion is matched by cutting the pooled
sample into ten equal-size strata. The analysis is run twice, once matching on
age and sex and once adding head motion, which quantifies how much of any group
difference is carried by residual motion. Deviation scores and raw features are
then compared through the same elastic net logistic regression (L1 and L2 weighted
equally, SAGA solver, balanced class weights) under ten-fold cross-validation,
reporting AUC-ROC, balanced accuracy, and F1. The difference in performance is
tested against a null built by permuting the labels and re-running the full
cross-validation for both feature sets.

## Notes on the implementation

- Imputation and scaling are fitted **inside** each cross-validation fold. Fitting
  them on the full dataset first would leak test-fold information into training
  and inflate the reported performance.
- The permutation null re-runs the whole cross-validation for both feature sets
  on each shuffle, so any advantage arising from the feature sets' differing
  dimensionality or conditioning is preserved under the null.
- Parcel quality control drops all-NaN, all-zero, and effectively flat time
  series before any metric is computed; a flat signal carries no phase
  information and would distort the local order parameter. Network definitions
  are remapped onto the surviving parcels rather than assuming a fixed index.
- The ELPD comparison works with both arviz 0.x and 1.x, which differ in whether
  `compare` takes an `ic` argument and in how they name the ELPD columns.

## Deviations from the preregistration

- The exclusion criterion for substance or behavioural addiction was dropped.
  It was collected in an online follow-up, and the missingness would have caused
  excessive data loss.
- Model comparison uses the expected log predictive density from leave-one-out
  cross-validation rather than the Akaike Information Criterion, to align with
  the Bayesian framework of the rest of the analysis.
- Head motion was added as a covariate to analyse its residual impact on the TD
  features after motion correction during preprocessing.

## References

- Deco, G., & Kringelbach, M. L. (2020). Turbulent-like dynamics in the human brain.
- Deco, G., et al. (2025). Turbulent dynamics across spatial scales.
- Mansour L., S., Di Biase, M. A., Smith, R. E., Zalesky, A., & Seguin, C. (2023).
  Connectomes for 40,000 UK Biobank participants.
- Rutherford, S., et al. (2023). Normative modelling of brain imaging data.
- Smith, D. J., et al. (2013). Prevalence and characteristics of probable major
  depression and bipolar disorder within UK Biobank.
- Sudlow, C., et al. (2015). UK Biobank: an open access resource.

## Licence

Released under the MIT Licence. The UK Biobank data itself is not covered by this
licence and must be obtained through UK Biobank.
