"""
Phase 2, part one: the Bayesian normative model.

A Bayesian linear regression is fitted per TD feature on the healthy control
training sample, predicting the feature from the covariates identified in Phase 1
(age, sex, and head motion). Posteriors are estimated with the No-U-Turn Sampler
through bambi's interface to PyMC, using bambi's default weakly informative
priors.

Candidate parameterisations are compared with the expected log predictive density
(ELPD) estimated by leave-one-out cross-validation, which tests whether b-splines
on the continuous covariates are warranted over a purely linear fit.

The fitted model is then applied to the Phase 2 cohort, and each participant's
deviation score is the z-score of their observed feature against the model's
predicted mean and residual standard deviation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

logging.getLogger("pymc").setLevel(logging.ERROR)

# Covariate roles, used to decide what gets scaled and what gets a spline basis.
CONTINUOUS_COVARIATES = [config.COL_AGE, config.COL_MOTION]
CATEGORICAL_COVARIATES = [config.COL_SEX]
COVARIATES = CONTINUOUS_COVARIATES + CATEGORICAL_COVARIATES


def candidate_formulas(outcome: str, spline_df: int = config.SPLINE_DF) -> Dict[str, str]:
    """
    The parameterisations compared by ELPD.

    Splines are considered for the continuous covariates only; sex is binary and
    enters linearly in every candidate.
    """
    age, motion, sex = config.COL_AGE, config.COL_MOTION, config.COL_SEX
    return {
        "linear": f"{outcome} ~ {age} + {sex} + {motion}",
        "spline_age_and_motion": (
            f"{outcome} ~ bs({age}, df={spline_df}) + {sex} "
            f"+ bs({motion}, df={spline_df})"
        ),
        "spline_motion_only": (
            f"{outcome} ~ {age} + {sex} + bs({motion}, df={spline_df})"
        ),
    }


@dataclass
class FittedNormativeModel:
    """A fitted model together with everything needed to apply it to new data."""

    outcome: str
    formula: str
    model: object
    idata: object
    scaler: StandardScaler
    scaled_columns: List[str] = field(default_factory=list)


def _prepare_training_frame(
    df: pd.DataFrame, outcome: str, covariates: Sequence[str] = COVARIATES
):
    """
    Select, coerce, and standardise the columns needed for one model.

    Returns the standardised frame, the fitted scaler, and the names of the
    columns the scaler was fitted on (the outcome plus continuous covariates).
    """
    columns = [outcome] + list(covariates)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for '{outcome}': {missing}")

    frame = df[columns].copy()

    numeric_columns = [outcome] + [
        c for c in CONTINUOUS_COVARIATES if c in covariates
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        raise ValueError(f"Too few complete cases to fit a model for '{outcome}'.")

    scaler = StandardScaler()
    scaled = frame.copy()
    scaled[numeric_columns] = scaler.fit_transform(frame[numeric_columns])

    return scaled, scaler, numeric_columns


def compare_parameterisations(
    df: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str] = COVARIATES,
    draws: int = config.NUTS_DRAWS,
    tune: int = config.NUTS_TUNE,
    chains: int = config.NUTS_CHAINS,
    **sample_kwargs,
) -> pd.DataFrame:
    """
    Compare the candidate parameterisations for one feature by ELPD-LOO.

    Returns arviz's comparison table, ranked best first. The log likelihood is
    retained during sampling because LOO needs it. Extra keyword arguments are
    passed through to the sampler (for example `cores=1`).
    """
    import arviz as az
    import bambi as bmb

    scaled, _, _ = _prepare_training_frame(df, outcome, covariates)

    traces = {}
    for name, formula in candidate_formulas(outcome).items():
        model = bmb.Model(formula, scaled)
        traces[name] = model.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
            **sample_kwargs,
        )

    return _compare_by_loo(az, traces, outcome)


def _compare_by_loo(az, traces: dict, var_name: str) -> pd.DataFrame:
    """
    Rank models by leave-one-out ELPD, tolerating both arviz APIs.

    arviz 0.x took an explicit `ic="loo"` argument and named its columns
    `elpd_loo` and `p_loo`; arviz 1.x dropped the argument (LOO is the only
    criterion) and renamed those columns to `elpd` and `p`. The column names are
    normalised to the 0.x spelling so the rest of the pipeline is unaffected.
    """
    try:
        comparison = az.compare(traces, ic="loo", var_name=var_name)
    except TypeError:
        comparison = az.compare(traces, var_name=var_name)

    return comparison.rename(columns={"elpd": "elpd_loo", "p": "p_loo"})


def compare_all_features(
    df: pd.DataFrame,
    outcomes: Sequence[str],
    covariates: Sequence[str] = COVARIATES,
    verbose: bool = True,
    **sample_kwargs,
) -> pd.DataFrame:
    """
    Run the ELPD comparison for several features and collect the results.

    Returns one row per feature and candidate, with the ELPD, its difference
    from the best model, and the stacking weight.
    """
    rows = []

    for outcome in outcomes:
        try:
            comparison = compare_parameterisations(
                df, outcome, covariates, **sample_kwargs
            )
        except Exception as error:  # a single feature failing should not stop the run
            print(f"Comparison failed for '{outcome}': {error}")
            continue

        if verbose:
            best = comparison.index[0]
            weight = comparison.iloc[0]["weight"]
            print(f"{outcome}: best = {best} (weight {weight:.2f})")

        for name, row in comparison.iterrows():
            rows.append(
                {
                    "outcome": outcome,
                    "parameterisation": name,
                    "rank": row["rank"],
                    "elpd_loo": row["elpd_loo"],
                    "elpd_diff": row["elpd_diff"],
                    "p_loo": row["p_loo"],
                    "weight": row["weight"],
                }
            )

    return pd.DataFrame(rows)


def fit_normative_model(
    train: pd.DataFrame,
    outcome: str,
    formula_key: str = "spline_motion_only",
    covariates: Sequence[str] = COVARIATES,
    draws: int = config.NUTS_DRAWS,
    tune: int = config.NUTS_TUNE,
    chains: int = config.NUTS_CHAINS,
    **sample_kwargs,
) -> FittedNormativeModel:
    """
    Fit the normative model for one feature on the healthy training sample.

    `formula_key` selects among the candidates compared by ELPD; pass the one
    that won the comparison for this dataset. Extra keyword arguments are passed
    through to the sampler (for example `cores=1`).
    """
    import bambi as bmb

    scaled, scaler, scaled_columns = _prepare_training_frame(
        train, outcome, covariates
    )
    formula = candidate_formulas(outcome)[formula_key]

    model = bmb.Model(formula, scaled)
    idata = model.fit(
        draws=draws, tune=tune, chains=chains, progressbar=False, **sample_kwargs
    )

    return FittedNormativeModel(
        outcome=outcome,
        formula=formula,
        model=model,
        idata=idata,
        scaler=scaler,
        scaled_columns=scaled_columns,
    )


def fit_all(
    train: pd.DataFrame,
    outcomes: Sequence[str],
    formula_key: str = "spline_motion_only",
    covariates: Sequence[str] = COVARIATES,
    verbose: bool = True,
    **sample_kwargs,
) -> Dict[str, FittedNormativeModel]:
    """Fit one normative model per feature, skipping any that fail."""
    fitted: Dict[str, FittedNormativeModel] = {}

    for outcome in outcomes:
        try:
            fitted[outcome] = fit_normative_model(
                train, outcome, formula_key, covariates, **sample_kwargs
            )
            if verbose:
                print(f"  fitted {outcome}")
        except Exception as error:
            print(f"  failed  {outcome}: {error}")

    return fitted


def deviation_scores(
    fitted: FittedNormativeModel,
    df: pd.DataFrame,
    covariates: Sequence[str] = COVARIATES,
) -> pd.Series:
    """
    Apply a fitted normative model to a cohort and return deviation z-scores.

    The posterior mean and residual standard deviation are predicted for each
    participant, both are returned to the original units of the feature, and the
    deviation is (observed - predicted mean) / predicted standard deviation.
    Participants with missing covariates receive NaN.
    """
    outcome = fitted.outcome
    columns = [outcome] + list(covariates)
    frame = df[columns].copy()

    for column in fitted.scaled_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Covariates must be complete to predict; a missing outcome only makes that
    # participant's own deviation undefined.
    usable = frame.dropna(subset=list(covariates))
    if usable.empty:
        return pd.Series(np.nan, index=df.index, name=f"{outcome}_deviation")

    scaled = usable.copy()
    # The outcome is part of the scaler's fitted columns; fill it so the
    # transform is well defined, then ignore the transformed value.
    scaled[outcome] = scaled[outcome].fillna(0.0)
    scaled[fitted.scaled_columns] = fitted.scaler.transform(
        scaled[fitted.scaled_columns]
    )

    posterior = fitted.model.predict(
        data=scaled, idata=fitted.idata, kind="response_params", inplace=False
    ).posterior

    mu_scaled = posterior["mu"].mean(dim=("chain", "draw")).to_numpy()
    sigma_scaled = posterior["sigma"].mean(dim=("chain", "draw")).to_numpy()

    # Undo the standardisation of the outcome so the deviation is on the
    # feature's own scale.
    outcome_position = fitted.scaled_columns.index(outcome)
    outcome_mean = fitted.scaler.mean_[outcome_position]
    outcome_sd = fitted.scaler.scale_[outcome_position]

    mu = mu_scaled * outcome_sd + outcome_mean
    sigma = sigma_scaled * outcome_sd

    observed = usable[outcome].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (observed - mu) / sigma

    result = pd.Series(np.nan, index=df.index, name=f"{outcome}_deviation")
    result.loc[usable.index] = z
    return result


def add_all_deviation_scores(
    fitted_models: Dict[str, FittedNormativeModel],
    df: pd.DataFrame,
    covariates: Sequence[str] = COVARIATES,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Append a deviation-score column for every fitted feature.

    Existing deviation columns with the same names are replaced.
    """
    deviations = {}

    for outcome, fitted in fitted_models.items():
        try:
            series = deviation_scores(fitted, df, covariates)
            deviations[series.name] = series
        except Exception as error:
            print(f"  deviation scores failed for {outcome}: {error}")

    deviation_frame = pd.DataFrame(deviations, index=df.index)
    overlapping = deviation_frame.columns.intersection(df.columns)
    if len(overlapping):
        df = df.drop(columns=list(overlapping))

    if verbose:
        print(f"Added {deviation_frame.shape[1]} deviation score columns.")

    return df.join(deviation_frame)
