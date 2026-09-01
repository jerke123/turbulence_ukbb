#!/usr/bin/env python
"""
Step 4 (Phase 2b): depressive symptom prediction.

Matches healthy controls to the depressive-symptom cohort, then asks whether
normative deviation scores classify group membership better than the raw,
uncorrected TD features.

The analysis is run twice: once with controls matched on age and sex, and once
with head motion added as a third matching criterion. Comparing the two
quantifies how much of any group difference is carried by residual head motion.

Outputs
-------
    results/phase2_performance.csv
    results/phase2_permutation_test.csv
    figures/phase2_roc_comparison.png/.svg
    figures/phase2_group_trajectories.png/.svg

Usage
-----
    python scripts/04_phase2_prediction.py
    python scripts/04_phase2_prediction.py --permutations 0   # skip the null
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, plotting, prediction  # noqa: E402
from src.cohort import (  # noqa: E402
    add_motion_bins,
    compare_covariates,
    load_psychiatric_exclusions,
    match_controls,
    select_depressive_symptoms,
    select_healthy_controls,
)

LABEL_COLUMN = "has_depressive_symptoms"
GROUP_LABEL = "group"


def deviation_columns_for(raw_columns: List[str], df: pd.DataFrame) -> List[str]:
    """Return the deviation column matching each raw feature, keeping them aligned."""
    pairs = [
        (raw, f"{raw}_deviation")
        for raw in raw_columns
        if f"{raw}_deviation" in df.columns
    ]
    if not pairs:
        raise ValueError(
            "No deviation columns found. Run 03_fit_normative_model.py first."
        )
    return [deviation for _, deviation in pairs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deviations", type=Path, default=config.DEVIATIONS_FILE)
    parser.add_argument("--icd10", type=Path, default=config.ICD10_FILE)
    parser.add_argument(
        "--permutations",
        type=int,
        default=config.N_PERMUTATIONS,
        help="Permutations for the null distribution; 0 skips the test "
             "(default: %(default)s).",
    )
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    np.random.seed(config.RANDOM_SEED)

    df = pd.read_csv(args.deviations)
    print(f"Loaded {len(df)} participants with deviation scores.\n")

    psychiatric = load_psychiatric_exclusions(args.icd10)
    controls = select_healthy_controls(df, psychiatric, verbose=False)
    cases = select_depressive_symptoms(df, verbose=False)
    print(f"{len(cases)} with depressive symptoms, {len(controls)} eligible controls.")

    # Head motion bins are cut on the pooled sample so both groups share edges.
    pooled = pd.concat([controls, cases], ignore_index=True)
    controls = add_motion_bins(controls, reference=pooled)
    cases = add_motion_bins(cases, reference=pooled)

    from src.phase1_lmm import FEATURES, find_scale_columns

    raw_columns: List[str] = []
    for feature in FEATURES:
        raw_columns.extend(find_scale_columns(df, feature))
    deviation_cols = deviation_columns_for(raw_columns, df)
    raw_columns = [c for c in raw_columns if f"{c}_deviation" in df.columns]
    print(f"Comparing {len(raw_columns)} raw features against "
          f"{len(deviation_cols)} deviation scores.\n")

    scenarios = {}
    performance_rows = []
    permutation_rows = []

    for title, matching_columns in [
        ("Matched on age and sex", [config.COL_AGE, config.COL_SEX]),
        (
            "Matched on age, sex, and head motion",
            [config.COL_AGE, config.COL_SEX, "motion_bin"],
        ),
    ]:
        print(f"{'=' * 70}\n{title}\n{'=' * 70}")

        matched = match_controls(cases, controls, matching_columns)
        analysis_set = pd.concat([matched, cases], ignore_index=True)
        analysis_set[LABEL_COLUMN] = (
            analysis_set[config.COL_RDS] > config.DEPRESSION_MIN_RDS
        ).astype(int)
        analysis_set[GROUP_LABEL] = np.where(
            analysis_set[LABEL_COLUMN] == 1, "Depressive symptoms", "Control"
        )

        residual = compare_covariates(
            analysis_set[analysis_set[LABEL_COLUMN] == 1],
            analysis_set[analysis_set[LABEL_COLUMN] == 0],
        )
        print("\nResidual covariate differences after matching:")
        print(residual.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

        deviation_result, raw_result, table = prediction.compare_feature_sets(
            analysis_set, raw_columns, deviation_cols, LABEL_COLUMN
        )
        table.insert(0, "scenario", title)
        table.insert(1, "n", len(analysis_set))
        performance_rows.append(table)

        print("\nCross-validated performance:")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        scenarios[title] = (deviation_result, raw_result)

        if args.permutations > 0:
            observed = np.mean(deviation_result.auc) - np.mean(raw_result.auc)
            print(f"\nPermutation test ({args.permutations} iterations)...")
            result = prediction.permutation_test(
                analysis_set[deviation_cols].to_numpy(dtype=float),
                analysis_set[raw_columns].to_numpy(dtype=float),
                analysis_set[LABEL_COLUMN].to_numpy(dtype=int),
                observed_difference=observed,
                n_permutations=args.permutations,
            )
            result["scenario"] = title
            permutation_rows.append(result)
            print(
                f"  observed AUC difference {result['observed_difference']:+.4f}, "
                f"p = {result['p_value']:.4f}"
            )

        print()

    pd.concat(performance_rows, ignore_index=True).to_csv(
        config.RESULTS_DIR / "phase2_performance.csv", index=False
    )
    if permutation_rows:
        pd.DataFrame(permutation_rows).to_csv(
            config.RESULTS_DIR / "phase2_permutation_test.csv", index=False
        )
    print(f"Wrote results to {config.RESULTS_DIR}")

    if not args.no_figure:
        print("Generating figures...")
        plotting.plot_roc_comparison(scenarios)

        plot_df = pd.concat([controls, cases], ignore_index=True)
        plot_df[GROUP_LABEL] = np.where(
            plot_df[config.COL_RDS] > config.DEPRESSION_MIN_RDS,
            "Depressive symptoms",
            "Control",
        )
        plotting.plot_group_trajectories(plot_df, GROUP_LABEL)
        print(f"Wrote figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
