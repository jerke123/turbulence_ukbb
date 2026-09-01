"""
Phase 1: healthy trends of turbulent dynamics across demographics.

For each TD feature, a linear mixed-effects model relates the feature to age,
sex, head motion, and spatial scale, with a random intercept per participant:

    feature ~ motion * C(scale) + age_centered * sex * C(scale) + (1 | subject)

Scale enters as a categorical factor so that interactions can be read at each
lambda and non-linear effects of scale are absorbed rather than assumed away.
An omnibus F-test is run per model, p-values are FDR-corrected, and partial eta
squared is reported as the effect size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

from . import config


@dataclass(frozen=True)
class Feature:
    """One TD feature and the column pattern that identifies its scale columns."""

    name: str
    label: str
    pattern: str


FEATURES = [
    Feature("amp_turb", "Amplitude turbulence", "amp_turb_0."),
    Feature("info_cascade_flow", "Information cascade flow", "info_cascade_flow_0."),
    Feature("info_transfer", "Information transfer", "info_transfer_0."),
]


def find_scale_columns(df: pd.DataFrame, feature: Feature) -> List[str]:
    """
    Return the whole-brain columns of a feature, one per spatial scale, in order.

    Network-specific and deviation-score columns are excluded: only columns whose
    name is the feature prefix followed directly by a lambda value are kept.
    """
    exact = re.compile(rf"^{re.escape(feature.name)}_\d+\.\d+$")
    columns = [c for c in df.columns if exact.match(c)]
    return sorted(columns, key=extract_scale)


def extract_scale(column: str) -> float:
    """Pull the lambda value out of a feature column name."""
    matches = re.findall(r"\d+\.\d+", column)
    if not matches:
        return float("nan")
    return float(matches[-1])


def to_long_format(
    df: pd.DataFrame,
    feature: Feature,
    id_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Reshape one feature from wide (one column per scale) to long format.

    Age is mean-centred so that the sex and scale main effects are interpretable
    at the sample's average age rather than at age zero.
    """
    if id_columns is None:
        id_columns = [
            config.COL_SUBJECT,
            config.COL_AGE,
            config.COL_SEX,
            config.COL_SITE,
            config.COL_MOTION,
        ]
    id_columns = [c for c in id_columns if c in df.columns]

    value_columns = find_scale_columns(df, feature)
    if not value_columns:
        raise ValueError(f"No columns found for feature '{feature.name}'.")

    long = pd.melt(
        df,
        id_vars=list(id_columns),
        value_vars=value_columns,
        var_name="column",
        value_name="value",
    )
    long["scale"] = long["column"].map(extract_scale)
    long = long.dropna(subset=["scale", "value"])
    long["age_centered"] = long[config.COL_AGE] - long[config.COL_AGE].mean()
    return long


def add_partial_eta_squared(anova: pd.DataFrame) -> pd.DataFrame:
    """
    Add partial eta squared to a Satterthwaite ANOVA table.

    For an F-test, partial eta squared = (F * df_num) / (F * df_num + df_den).
    """
    anova = anova.copy()
    numerator = anova["F-stat"] * anova["NumDF"]
    anova["partial_eta_sq"] = numerator / (numerator + anova["DenomDF"])
    return anova


def interpret_effect_size(eta_squared: float) -> str:
    """Label an effect size against Cohen's benchmarks."""
    if pd.isna(eta_squared):
        return "n/a"
    if eta_squared >= config.ETA_SQ_LARGE:
        return "large"
    if eta_squared >= config.ETA_SQ_MEDIUM:
        return "medium"
    if eta_squared >= config.ETA_SQ_SMALL:
        return "small"
    return "negligible"


def add_fdr(table: pd.DataFrame, p_column: str = "P-val") -> pd.DataFrame:
    """FDR-correct a column of p-values, adding q-values and a significance flag."""
    table = table.copy()
    p_values = table[p_column]
    valid = p_values.notna()

    table["q_val"] = np.nan
    table["significant_fdr"] = False

    if valid.any():
        rejected, q_values = fdrcorrection(
            p_values[valid].to_numpy(), alpha=config.FDR_ALPHA
        )
        table.loc[valid, "q_val"] = q_values
        table.loc[valid, "significant_fdr"] = rejected

    return table


def fit_feature_lmm(
    long: pd.DataFrame,
    outcome: str = "value",
    formula: str | None = None,
):
    """
    Fit the mixed-effects model for one feature and return its result tables.

    Uses pymer4's interface to lme4, which supplies Satterthwaite degrees of
    freedom for the omnibus F-tests.

    Returns
    -------
    (anova, coefficients)
        The omnibus ANOVA table with partial eta squared and FDR-corrected
        q-values, and the fixed-effect coefficient table with FDR-corrected
        q-values.
    """
    from pymer4.models import Lmer
    from rpy2.robjects import conversion, default_converter

    if formula is None:
        formula = config.LMM_FORMULA.format(
            outcome=outcome,
            motion=config.COL_MOTION,
            sex=config.COL_SEX,
            subject=config.COL_SUBJECT,
        )

    if long["scale"].nunique() < 2:
        raise ValueError("At least two spatial scales are needed to fit the model.")

    with conversion.localconverter(default_converter):
        model = Lmer(formula, data=long)
        model.fit(summarize=False)
        coefficients = model.coefs.copy()
        anova = model.anova().copy()

    anova = add_fdr(add_partial_eta_squared(anova))
    anova["effect_size"] = anova["partial_eta_sq"].map(interpret_effect_size)
    coefficients = add_fdr(coefficients)

    return anova, coefficients


def run_phase1(
    df: pd.DataFrame,
    features: Sequence[Feature] = FEATURES,
    verbose: bool = True,
):
    """
    Fit the Phase 1 model for every TD feature.

    Returns
    -------
    (anova_tables, coefficient_tables)
        Two dictionaries keyed by feature name.
    """
    anova_tables, coefficient_tables = {}, {}

    for feature in features:
        if verbose:
            print(f"\n{'=' * 70}\n{feature.label}\n{'=' * 70}")

        try:
            long = to_long_format(df, feature)
        except ValueError as error:
            print(f"  skipped: {error}")
            continue

        anova, coefficients = fit_feature_lmm(long)
        anova_tables[feature.name] = anova
        coefficient_tables[feature.name] = coefficients

        if verbose:
            columns = ["F-stat", "P-val", "q_val", "partial_eta_sq", "effect_size"]
            print(anova[columns].to_string(float_format=lambda v: f"{v:.4f}"))

    return anova_tables, coefficient_tables


def scale_wise_slopes(
    long: pd.DataFrame, predictor: str, standardise: bool = True
) -> pd.DataFrame:
    """
    Estimate a simple per-scale slope of the feature on a continuous predictor.

    This is descriptive only, used for the sensitivity panels in the figures; the
    inferential results come from the mixed-effects model above. Values are
    standardised within scale first so that slopes are comparable across scales
    that differ in magnitude.
    """
    from scipy import stats

    long = long.copy()
    if standardise:
        long["value"] = long.groupby("scale")["value"].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    rows = []
    for scale, group in long.groupby("scale"):
        subset = group.dropna(subset=[predictor, "value"])
        if len(subset) < 2:
            continue
        slope = stats.linregress(subset[predictor], subset["value"]).slope
        rows.append({"scale": scale, "slope": slope})

    return pd.DataFrame(rows).sort_values("scale").reset_index(drop=True)


def scale_wise_group_difference(
    long: pd.DataFrame, group_column: str = config.COL_SEX
) -> pd.DataFrame:
    """
    Standardised difference between two groups at each spatial scale.

    Groups are ordered alphabetically and the difference is second minus first.
    """
    long = long.copy()
    long["value"] = long.groupby("scale")["value"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    rows = []
    for scale, group in long.groupby("scale"):
        subset = group.dropna(subset=[group_column, "value"])
        levels = sorted(subset[group_column].unique())
        if len(levels) != 2:
            continue
        difference = (
            subset.loc[subset[group_column] == levels[1], "value"].mean()
            - subset.loc[subset[group_column] == levels[0], "value"].mean()
        )
        rows.append(
            {
                "scale": scale,
                "difference": difference,
                "reference": levels[0],
                "comparison": levels[1],
            }
        )

    return pd.DataFrame(rows).sort_values("scale").reset_index(drop=True)
