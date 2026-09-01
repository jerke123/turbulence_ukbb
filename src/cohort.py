"""
Cohort construction.

Turns the raw phenotype export into the analysis samples described in the
Methods: an overall eligible sample, a strictly screened healthy control (HC)
cohort for Phase 1, and a depressive-symptom cohort with matched controls for
Phase 2.

The derived variables and exclusions are:

  rds                 Recent Depressive Symptoms, summed over four items (4-16)
  bipolar             self-reported mania/hypomania screen
  antidepressant_use  current SSRI/SNRI from self-reported medication
  depression_group    HC / Single ep. / Moderate / Severe / Bipolar
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from . import config
from .config import Fields

# The four RDS items are scored on a four-point frequency scale.
RDS_RESPONSE_SCORES = {
    "Not at all": 1,
    "Several days": 2,
    "More than half the days": 3,
    "Nearly every day": 4,
    "Do not know": np.nan,
    "Prefer not to answer": np.nan,
}

MISSING_RESPONSES = ["Do not know", "Prefer not to answer"]

SEVERITY_ORDER = ["HC", "Single ep.", "Moderate", "Severe", "Bipolar"]


# --------------------------------------------------------------------------- #
# Derived variables
# --------------------------------------------------------------------------- #
def add_rds_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum the four RDS items into a single score.

    Participants missing any item are dropped, since a partial sum is not
    comparable to a complete one.
    """
    df = df.copy()
    items = df[Fields.RDS_ITEMS].replace(RDS_RESPONSE_SCORES)
    df[config.COL_RDS] = items.sum(axis=1, min_count=len(Fields.RDS_ITEMS))
    return df.dropna(subset=[config.COL_RDS])


def add_bipolar_screen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag probable bipolar disorder from the self-reported mania screen.

    Following the UK Biobank self-report guidelines (Smith et al., 2013), a
    participant screens positive when they report a period of elevated or
    irritable mood, endorse at least three of four manic symptoms, and report
    that the period lasted a week or more. Ambiguous responses are treated
    conservatively as positive so that they are excluded rather than retained.
    """
    df = df.copy()
    mood_symptoms = ["more active", "more talkative", "less sleep", "more ideas"]

    elevated_mood = (
        df[Fields.MANIA_EVER_2DAYS].isin(["Yes"] + MISSING_RESPONSES)
        | df[Fields.IRRITABLE_EVER_2DAYS].isin(["Yes"] + MISSING_RESPONSES)
    )

    symptoms = df[Fields.MANIA_SYMPTOMS].fillna("")
    symptom_count = sum(
        symptoms.str.contains(term, case=False, regex=False).astype(int)
        for term in mood_symptoms
    )
    enough_symptoms = (symptom_count > 2) | (symptoms == "All of the above")

    long_enough = df[Fields.MANIA_DURATION].isin(
        ["A week or more"] + MISSING_RESPONSES
    )

    df["bipolar"] = (elevated_mood & enough_symptoms & long_enough).astype(int)
    return df


def add_antidepressant_use(
    df: pd.DataFrame, drug_names: Sequence[str] = config.SSRI_SNRI_NAMES
) -> pd.DataFrame:
    """
    Flag current SSRI/SNRI use from the free-text medication fields, and record
    whether the participant reported that antidepressants helped them.

    `antidepressant_response` is 1 (helped), 0 (did not help), or -1 (item never
    presented, i.e. the participant was not asked about antidepressants).
    """
    df = df.copy()

    medication_columns = [
        column
        for column in df.columns
        if column.startswith(Fields.MEDICATION_PREFIX)
    ]
    if medication_columns:
        pattern = r"\b(?:" + "|".join(drug_names) + r")\b"
        found = (
            df[medication_columns]
            .stack()
            .str.contains(pattern, case=False, na=False)
            .groupby(level=0)
            .any()
            .reindex(df.index, fill_value=False)
        )
        df["antidepressant_use"] = found.astype(int)
    else:
        df["antidepressant_use"] = 0

    response_columns = [c for c in Fields.ANTIDEPRESSANT_RESPONSE if c in df.columns]
    if response_columns:
        df["antidepressant_response"] = np.select(
            [
                (df[response_columns] == "Yes, at least a little").any(axis=1),
                (df[response_columns] == "No").any(axis=1),
            ],
            [1, 0],
            default=-1,
        )
    else:
        df["antidepressant_response"] = -1

    return df


def _to_numeric_response(series: pd.Series) -> pd.Series:
    """Coerce a self-report count field to an integer, with -1 for missing."""
    return pd.to_numeric(
        series.replace(MISSING_RESPONSES + [np.nan], -1), errors="coerce"
    ).fillna(-1).astype(int)


def add_depression_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each participant to a depression recurrence-severity category.

    HC requires no current symptoms, no help-seeking for psychological distress,
    no prolonged low mood, and no antidepressant history. The depressed
    categories require help-seeking plus a low-mood episode of at least two
    weeks, and are then split by the number of reported episodes.
    """
    df = df.copy()
    df[Fields.LONGEST_EPISODE_WEEKS] = _to_numeric_response(
        df[Fields.LONGEST_EPISODE_WEEKS]
    )
    df[Fields.N_EPISODES] = _to_numeric_response(df[Fields.N_EPISODES])

    sought_help = (df[Fields.SEEN_GP_FOR_NERVES] == "Yes") | (
        df[Fields.SEEN_PSYCHIATRIST_FOR_NERVES] == "Yes"
    )
    prolonged_episode = df[Fields.LONGEST_EPISODE_WEEKS] >= 2
    episodes = df[Fields.N_EPISODES]

    group = pd.Series(np.nan, index=df.index, dtype=object)

    group[
        (df[config.COL_RDS] == config.HC_MAX_RDS)
        & (df[Fields.SEEN_GP_FOR_NERVES] == "No")
        & (df[Fields.SEEN_PSYCHIATRIST_FOR_NERVES] == "No")
        & (df[Fields.LONGEST_EPISODE_WEEKS] < 2)
        & (df["antidepressant_response"] == -1)
        & (df["antidepressant_use"] == 0)
    ] = "HC"

    group[sought_help & prolonged_episode & (episodes == 1)] = "Single ep."
    group[sought_help & prolonged_episode & (episodes > 1) & (episodes <= 5)] = (
        "Moderate"
    )
    group[sought_help & prolonged_episode & (episodes > 5)] = "Severe"

    if "bipolar" in df.columns:
        group[df["bipolar"] == 1] = "Bipolar"

    df[config.COL_GROUP] = pd.Categorical(
        group, categories=SEVERITY_ORDER, ordered=True
    )
    return df


# --------------------------------------------------------------------------- #
# Loading and eligibility
# --------------------------------------------------------------------------- #
def load_phenotypes(path=None) -> pd.DataFrame:
    """
    Read the phenotype export and derive every variable the analysis needs.

    Renames the UK Biobank field identifiers to readable names and restricts to
    participants with a usable first imaging visit.
    """
    path = path or config.PHENOTYPE_FILE
    df = pd.read_csv(path)

    required = [Fields.SEX, Fields.AGE_IMAGING, Fields.CORTICAL_SCAN_ID]
    df = df.dropna(subset=required).copy()

    # The scan identifier carries the participant id as its first component.
    df[config.COL_SUBJECT] = (
        df[Fields.CORTICAL_SCAN_ID].astype(str).str.split("_").str[0]
    )

    df = df.rename(
        columns={
            Fields.SEX: config.COL_SEX,
            Fields.AGE_IMAGING: config.COL_AGE,
            Fields.SITE: config.COL_SITE,
            Fields.MEAN_REL_HEAD_MOTION: config.COL_MOTION,
            Fields.DVARS: config.COL_DVARS,
        }
    )
    df[config.COL_AGE] = pd.to_numeric(df[config.COL_AGE], errors="coerce")

    df = add_rds_score(df)
    df = add_bipolar_screen(df)
    df = add_antidepressant_use(df)
    df = add_depression_group(df)

    return df.reset_index(drop=True)


def load_neurological_exclusions(path=None) -> set:
    """
    Return the participant ids with any ICD-10 Chapter VI diagnosis (G00-G99),
    i.e. a disease of the nervous system.
    """
    path = path or config.ICD10_FILE
    icd = pd.read_csv(path)
    icd[config.COL_SUBJECT] = (
        icd[Fields.CORTICAL_SCAN_ID].astype(str).str.split("_").str[0]
    )
    has_g_code = (
        icd[Fields.ICD10].astype(str).str.contains(r"\bG\d{2}", regex=True, na=False)
    )
    return set(icd.loc[has_g_code, config.COL_SUBJECT].unique())


def load_psychiatric_exclusions(path=None) -> set:
    """
    Return the participant ids with any ICD-10 Chapter V diagnosis (F00-F99),
    i.e. a mental or behavioural disorder.
    """
    path = path or config.ICD10_FILE
    icd = pd.read_csv(path)
    icd[config.COL_SUBJECT] = (
        icd[Fields.CORTICAL_SCAN_ID].astype(str).str.split("_").str[0]
    )
    has_f_code = (
        icd[Fields.ICD10].astype(str).str.contains(r"\bF\d{2}", regex=True, na=False)
    )
    return set(icd.loc[has_f_code, config.COL_SUBJECT].unique())


def apply_common_exclusions(
    df: pd.DataFrame, neurological_ids: Iterable[str], verbose: bool = True
) -> pd.DataFrame:
    """
    Apply the exclusions shared by both analysis phases.

    Removes participants with a nervous system disease, a positive bipolar
    screen, or a scan from an excluded site, printing the count lost at each
    step so the numbers can be checked against the participant flowchart.
    """
    steps = []

    before = len(df)
    df = df[~df[config.COL_SUBJECT].isin(set(neurological_ids))]
    steps.append(("nervous system disease (ICD-10 G00-G99)", before - len(df)))

    before = len(df)
    df = df[df[config.COL_GROUP] != "Bipolar"]
    steps.append(("self-reported bipolar disorder", before - len(df)))

    before = len(df)
    df = df[~df[config.COL_SITE].isin(config.EXCLUDED_SITES)]
    steps.append(("excluded scanning site", before - len(df)))

    if verbose:
        print("Common exclusions:")
        for reason, n in steps:
            print(f"  -{n:>6}  {reason}")
        print(f"  ={len(df):>7}  eligible sample")

    return df.reset_index(drop=True)


def select_healthy_controls(
    df: pd.DataFrame, psychiatric_ids: Iterable[str], verbose: bool = True
) -> pd.DataFrame:
    """
    Build the Phase 1 healthy control cohort.

    Strict criteria: no depressive symptoms above the RDS floor, no self-reported
    depression history or help-seeking, and no ICD-10 mental or behavioural
    disorder diagnosis.
    """
    steps = []

    before = len(df)
    hc = df[df[config.COL_RDS] <= config.HC_MAX_RDS]
    steps.append((f"RDS > {config.HC_MAX_RDS}", before - len(hc)))

    before = len(hc)
    hc = hc[hc[config.COL_GROUP] == "HC"]
    steps.append(("self-reported depressive symptoms", before - len(hc)))

    before = len(hc)
    hc = hc[~hc[config.COL_SUBJECT].isin(set(psychiatric_ids))]
    steps.append(("mental or behavioural disorder (ICD-10 F00-F99)", before - len(hc)))

    if verbose:
        print("\nPhase 1 healthy control cohort:")
        for reason, n in steps:
            print(f"  -{n:>6}  {reason}")
        print(f"  ={len(hc):>7}  healthy controls")

    return hc.reset_index(drop=True)


def select_depressive_symptoms(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build the Phase 2 depressive-symptom cohort.

    Requires current symptoms above the RDS cut-off together with having
    consulted a professional for nerves, anxiety, or depression.
    """
    before = len(df)
    consulted_professional = (df[Fields.SEEN_GP_FOR_NERVES] == "Yes") | (
        df[Fields.SEEN_PSYCHIATRIST_FOR_NERVES] == "Yes"
    )
    cases = df[(df[config.COL_RDS] > config.DEPRESSION_MIN_RDS) & consulted_professional]

    if verbose:
        print("\nPhase 2 depressive symptom cohort:")
        print(f"  -{before - len(cases):>6}  RDS <= {config.DEPRESSION_MIN_RDS} "
              f"or no professional consultation")
        print(f"  ={len(cases):>7}  participants with depressive symptoms")

    return cases.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Covariate comparison and matching
# --------------------------------------------------------------------------- #
def compare_covariates(
    cases: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    """
    Test whether the covariates of interest differ between the two groups.

    Uses Fisher's exact test for sex, Welch's t-test for age, and a Mann-Whitney
    U test for head motion, reporting Hedges' g or Cramer's V as effect sizes.
    """
    from scipy import stats

    def hedges_g(a: pd.Series, b: pd.Series) -> float:
        n1, n2 = len(a), len(b)
        pooled_sd = np.sqrt(
            ((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2)
        )
        d = (a.mean() - b.mean()) / pooled_sd
        return d * (1 - 3 / (4 * (n1 + n2) - 9))  # small-sample correction

    rows = []

    age_cases = pd.to_numeric(cases[config.COL_AGE], errors="coerce").dropna()
    age_controls = pd.to_numeric(controls[config.COL_AGE], errors="coerce").dropna()
    t_stat, p_age = stats.ttest_ind(age_cases, age_controls, equal_var=False)
    rows.append(
        {
            "covariate": "age",
            "test": "Welch's t-test",
            "statistic": t_stat,
            "p": p_age,
            "effect_size": hedges_g(age_cases, age_controls),
            "effect_size_name": "Hedges' g",
        }
    )

    # Build the table from plain arrays: the two frames may carry overlapping
    # index labels, which would break index-aligned concatenation.
    group_labels = np.concatenate(
        [np.repeat("case", len(cases)), np.repeat("control", len(controls))]
    )
    sex_values = np.concatenate(
        [
            cases[config.COL_SEX].to_numpy(dtype=object),
            controls[config.COL_SEX].to_numpy(dtype=object),
        ]
    )
    contingency = pd.crosstab(pd.Series(group_labels), pd.Series(sex_values))
    odds_ratio, p_sex = stats.fisher_exact(contingency)
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    cramers_v = np.sqrt(
        (chi2 / n) / min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    )
    rows.append(
        {
            "covariate": "sex",
            "test": "Fisher's exact",
            "statistic": odds_ratio,
            "p": p_sex,
            "effect_size": cramers_v,
            "effect_size_name": "Cramer's V",
        }
    )

    motion_cases = pd.to_numeric(cases[config.COL_MOTION], errors="coerce").dropna()
    motion_controls = pd.to_numeric(
        controls[config.COL_MOTION], errors="coerce"
    ).dropna()
    u_stat, p_motion = stats.mannwhitneyu(
        motion_cases, motion_controls, alternative="two-sided"
    )
    rows.append(
        {
            "covariate": "head motion",
            "test": "Mann-Whitney U",
            "statistic": u_stat,
            "p": p_motion,
            "effect_size": hedges_g(motion_cases, motion_controls),
            "effect_size_name": "Hedges' g",
        }
    )

    return pd.DataFrame(rows)


def add_motion_bins(
    df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    n_bins: int = config.N_MOTION_BINS,
    column: str = config.COL_MOTION,
) -> pd.DataFrame:
    """
    Assign each participant to one of `n_bins` equal-size head-motion strata.

    Bin edges are taken from `reference` (the pooled sample by default) and the
    outer edges are extended to infinity, so that cases with more or less motion
    than any control still fall into a stratum rather than becoming unmatched.
    """
    df = df.copy()
    source = reference if reference is not None else df

    edges = pd.qcut(
        pd.to_numeric(source[column], errors="coerce"),
        q=n_bins,
        retbins=True,
        duplicates="drop",
    )[1]
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf

    df["motion_bin"] = pd.cut(
        pd.to_numeric(df[column], errors="coerce"), bins=edges
    )
    return df


def match_controls(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    matching_columns: Sequence[str],
    random_state: int = config.RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Draw a control group with the same joint distribution as the cases.

    For every stratum defined by `matching_columns`, sample as many controls as
    there are cases (or as many as exist, if the pool is smaller). Returns the
    matched controls only; concatenate with `cases` to obtain the analysis set.
    """
    required_per_stratum = cases.groupby(
        list(matching_columns), observed=True
    ).size()

    shortfalls: List[str] = []

    def sample_stratum(group: pd.DataFrame) -> pd.DataFrame | None:
        stratum = group.name
        required = int(required_per_stratum.get(stratum, 0))
        if required == 0:
            return None
        n = min(required, len(group))
        if n < required:
            shortfalls.append(f"{stratum}: needed {required}, sampled {n}")
        return group.sample(n=n, random_state=random_state)

    matched = controls.groupby(
        list(matching_columns), observed=True, group_keys=False
    ).apply(sample_stratum)

    if verbose:
        print(
            f"Matched on {', '.join(matching_columns)}: "
            f"{len(matched)} controls for {len(cases)} cases"
        )
        for message in shortfalls[:10]:
            print(f"  under-matched stratum {message}")
        if len(shortfalls) > 10:
            print(f"  ... and {len(shortfalls) - 10} further strata")

    return matched.reset_index(drop=True)
