"""
Central configuration for the turbulent dynamics (TD) normative modelling pipeline.

Everything that is a study-level choice (spatial scales, filter band, cohort
cut-offs, model settings) lives here so that the analysis scripts stay readable
and a reviewer can check the parameters against the Methods section in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# All paths can be overridden with environment variables so that no local or
# platform-specific location is hard-coded into the analysis code.

DATA_DIR = Path(os.environ.get("TD_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("TD_RESULTS_DIR", "./results"))
FIGURES_DIR = Path(os.environ.get("TD_FIGURES_DIR", "./figures"))

# Subject-level phenotype table exported from the cohort browser (one row per
# participant, containing the UK Biobank fields listed under `fields` below).
PHENOTYPE_FILE = DATA_DIR / "phenotypes.csv"

# Table of ICD-10 diagnoses (long or wide) used for the neurological and
# psychiatric exclusions.
ICD10_FILE = DATA_DIR / "icd10.csv"

# Output of `scripts/01_compute_td_features.py`: phenotypes + TD features.
FEATURES_FILE = RESULTS_DIR / "td_features.csv"

# Output of `scripts/03_fit_normative_model.py`: features + deviation z-scores.
DEVIATIONS_FILE = RESULTS_DIR / "td_features_with_deviations.csv"


# --------------------------------------------------------------------------- #
# Acquisition and preprocessing
# --------------------------------------------------------------------------- #
TR = 0.735  # repetition time of the UK Biobank rs-fMRI protocol, in seconds

# Band-pass filter applied to the parcellated BOLD time series before the
# Hilbert transform.
BANDPASS_LOW_HZ = 0.008
BANDPASS_HIGH_HZ = 0.08
BANDPASS_ORDER = 4

# Regions whose time series are all-NaN, all-zero, or effectively flat are
# dropped before any turbulence metric is computed.
ROI_STD_THRESHOLD = 1e-3


# --------------------------------------------------------------------------- #
# Turbulent dynamics: spatial scales
# --------------------------------------------------------------------------- #
# The inverse spatial-scale parameter lambda of the exponential distance rule.
# 11 steps from 0.01 to 0.31 correspond to spatial scales of roughly 100 mm
# down to 3 mm.
N_LAMBDA_STEPS = 11
LAMBDA_MIN = 0.01
LAMBDA_MAX = 0.31
LAMBDA_SCALES = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_LAMBDA_STEPS)

# Lag (in volumes) used when correlating a fine scale at t + dt against the
# adjacent coarser scale at t for the information cascade flow.
CASCADE_FLOW_LAG = 6

# Distance binning used for the information transfer log-log fit.
TRANSFER_N_BINS = 400
TRANSFER_FIT_FIRST_BIN = 20
TRANSFER_FIT_LAST_BIN = 80


# --------------------------------------------------------------------------- #
# Parcellation
# --------------------------------------------------------------------------- #
N_CORTICAL_PARCELS = 1000  # Schaefer 2018, 7 networks
N_SUBCORTICAL_PARCELS = 54  # Tian Subcortex, Scale IV, 3T
YEO_NETWORKS = 7

SCHAEFER_CENTROID_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/"
    "Centroid_coordinates/"
    "Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
)
TIAN_CENTROID_URL = (
    "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
    "Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S4_3T_COG.txt"
)

# File names inside the per-subject imaging archives.
CORTICAL_TIMESERIES_FILE = "fMRI.Schaefer7n1000p.csv.gz"
SUBCORTICAL_TIMESERIES_FILE = "fMRI.Tian_Subcortex_S4_3T.csv.gz"


# --------------------------------------------------------------------------- #
# UK Biobank field identifiers
# --------------------------------------------------------------------------- #
# Named here once so that the cohort code reads in terms of concepts rather
# than field numbers. Suffix `_i2` denotes the first imaging visit.
class Fields:
    SEX = "p31"
    AGE_IMAGING = "p21003_i2"
    SITE = "p54"
    MEAN_REL_HEAD_MOTION = "p24441"  # mean framewise displacement
    DVARS = "p24432"

    # Recent Depressive Symptoms (RDS-4): the four PHQ-derived items.
    RDS_ITEMS = ["p2050", "p2060", "p2070", "p2080"]

    # Help-seeking for psychological distress.
    SEEN_GP_FOR_NERVES = "p2090"
    SEEN_PSYCHIATRIST_FOR_NERVES = "p2100"

    # Self-reported depressive episode characteristics.
    LONGEST_EPISODE_WEEKS = "p4609"
    N_EPISODES = "p4620"
    LONGEST_EPISODE_ALT = "p5375"

    # Self-reported mania/hypomania screen, used for the bipolar exclusion.
    MANIA_EVER_2DAYS = "p4642"
    IRRITABLE_EVER_2DAYS = "p4653"
    MANIA_SYMPTOMS = "p6156"
    MANIA_DURATION = "p5663"

    # Current medication (self-reported, imaging visit) and antidepressant
    # response items.
    MEDICATION_PREFIX = "p20003_i2"
    ANTIDEPRESSANT_RESPONSE = ["p29040", "p29041", "p29042", "p29043", "p29044", "p29045"]

    # ICD-10 diagnosis array.
    ICD10 = "p41270"

    # Identifiers linking the phenotype table to the imaging archives.
    CORTICAL_SCAN_ID = "p31018_i2"
    SUBCORTICAL_SCAN_ID = "p31019_i2"


# Readable names used throughout the analysis after renaming.
COL_AGE = "age"
COL_SEX = "sex"
COL_SITE = "site"
COL_MOTION = "mean_rel_head_motion"
COL_DVARS = "dvars"
COL_RDS = "rds"
COL_GROUP = "depression_group"
COL_SUBJECT = "eid"


# --------------------------------------------------------------------------- #
# Cohort definitions
# --------------------------------------------------------------------------- #
# RDS-4 is scored 1-4 per item, so 4 is the floor (no symptoms at all).
RDS_MIN = 4

# Phase 1 healthy controls: no depressive symptoms beyond the floor.
HC_MAX_RDS = 4

# Phase 2 depressive-symptom group: RDS > 8, in line with a PHQ-9 equivalent
# risk of at least a mild depressive episode.
DEPRESSION_MIN_RDS = 8

# Site excluded because of its small participant count.
EXCLUDED_SITES = ["Bristol (imaging)"]

# Antidepressants screened for in the self-reported medication fields.
SSRI_SNRI_NAMES = [
    # SSRIs
    "citalopram", "cipramil",
    "dapoxetine", "priligy",
    "escitalopram", "cipralex", "lexapro",
    "fluoxetine", "prozac", "olena",
    "fluvoxamine", "faverin",
    "paroxetine", "seroxat",
    "sertraline", "lustral",
    "vortioxetine", "brintellix",
    # SNRIs
    "duloxetine", "cymbalta", "yentreve",
    "venlafaxine", "efexor",
    "desvenlafaxine", "pristiq",
]


# --------------------------------------------------------------------------- #
# Phase 1: linear mixed-effects models
# --------------------------------------------------------------------------- #
# Scale is entered as a categorical factor so that non-linear effects across
# lambda are captured and interactions can be read per scale.
LMM_FORMULA = (
    "{outcome} ~ {motion} * C(scale) + age_centered * {sex} * C(scale) + (1|{subject})"
)
FDR_ALPHA = 0.05

# Cohen's benchmarks for partial eta squared, used when reporting effect sizes.
ETA_SQ_SMALL = 0.01
ETA_SQ_MEDIUM = 0.06
ETA_SQ_LARGE = 0.14


# --------------------------------------------------------------------------- #
# Phase 2: Bayesian normative model
# --------------------------------------------------------------------------- #
NUTS_DRAWS = 2000
NUTS_TUNE = 1000
NUTS_CHAINS = 4
SPLINE_DF = 4  # degrees of freedom for the b-spline basis on continuous covariates


# --------------------------------------------------------------------------- #
# Phase 2: depressive symptom prediction
# --------------------------------------------------------------------------- #
N_MOTION_BINS = 10  # equal-size head-motion strata used for matching
CV_FOLDS = 10
N_PERMUTATIONS = 1000
ELASTIC_NET_L1_RATIO = 0.5  # L1 and L2 weighted equally
ELASTIC_NET_C = 1.0
MAX_ITER = 10_000
RANDOM_SEED = 42


def ensure_output_dirs() -> None:
    """Create the results and figures directories if they do not exist yet."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
