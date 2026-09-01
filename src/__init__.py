"""
Turbulent dynamics and normative modelling of depressive symptoms.

Modules
-------
config            study-level parameters and field identifiers
parcellation      parcel centroids, distance matrix, network index, ROI QC
turbulence        the three turbulent dynamics features
io_timeseries     loading parcellated BOLD time series
cohort            cohort derivation, exclusions, covariate tests, matching
phase1_lmm        mixed-effects models of demographic and motion effects
normative_model   Bayesian normative model and deviation scores
prediction        classification, cross-validation, permutation testing
plotting          figures
"""

__all__ = [
    "config",
    "parcellation",
    "turbulence",
    "io_timeseries",
    "cohort",
    "phase1_lmm",
    "normative_model",
    "prediction",
    "plotting",
]
