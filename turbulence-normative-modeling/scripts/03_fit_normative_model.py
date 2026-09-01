#!/usr/bin/env python
"""
Step 3 (Phase 2a): fit the Bayesian normative model and derive deviation scores.

Two stages:

  --compare   Fit the candidate parameterisations (linear, splines on age and
              motion, splines on motion only) on a subset of features and rank
              them by ELPD from leave-one-out cross-validation. Run this first
              and read off which parameterisation wins.

  (default)   Fit the chosen parameterisation for every feature on the healthy
              control training sample, then apply it to the full cohort and
              append one deviation score column per feature.

Outputs
-------
    results/normative_model_comparison.csv
    results/td_features_with_deviations.csv

Usage
-----
    python scripts/03_fit_normative_model.py --compare
    python scripts/03_fit_normative_model.py --parameterisation spline_motion_only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, normative_model  # noqa: E402
from src.cohort import (  # noqa: E402
    apply_common_exclusions,
    load_neurological_exclusions,
    load_psychiatric_exclusions,
    select_healthy_controls,
)


def find_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Collect the whole-brain TD feature columns to model.

    Matches `<feature>_<lambda>` exactly, so network-specific columns, the
    across-scale means, and any existing deviation scores are left out.
    """
    prefixes = ["amp_turb", "info_transfer", "info_cascade_flow"]
    pattern = re.compile(rf"^({'|'.join(prefixes)})_\d+\.\d+$")
    return sorted(c for c in df.columns if pattern.match(c))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.FEATURES_FILE)
    parser.add_argument("--icd10", type=Path, default=config.ICD10_FILE)
    parser.add_argument("--output", type=Path, default=config.DEVIATIONS_FILE)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare parameterisations by ELPD instead of fitting the model.",
    )
    parser.add_argument(
        "--compare-n",
        type=int,
        default=3,
        help="How many features to include in the comparison; sampling a few is "
             "enough to choose a parameterisation (default: %(default)s).",
    )
    parser.add_argument(
        "--parameterisation",
        default="spline_motion_only",
        choices=["linear", "spline_age_and_motion", "spline_motion_only"],
        help="Model form to fit (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} participants.")

    neurological = load_neurological_exclusions(args.icd10)
    psychiatric = load_psychiatric_exclusions(args.icd10)

    eligible = apply_common_exclusions(df, neurological, verbose=False)
    training = select_healthy_controls(eligible, psychiatric, verbose=False)
    print(f"Normative training sample: {len(training)} healthy controls.")

    outcomes = find_feature_columns(df)
    print(f"Modelling {len(outcomes)} TD features.\n")

    if args.compare:
        subset = outcomes[: args.compare_n]
        print(f"Comparing parameterisations on: {', '.join(subset)}\n")
        comparison = normative_model.compare_all_features(training, subset)
        path = config.RESULTS_DIR / "normative_model_comparison.csv"
        comparison.to_csv(path, index=False)
        print(f"\nWrote the ELPD comparison to {path}")
        print("Re-run with --parameterisation set to the winner to fit the model.")
        return

    print(f"Fitting the '{args.parameterisation}' model for each feature...")
    fitted = normative_model.fit_all(
        training, outcomes, formula_key=args.parameterisation
    )
    print(f"Fitted {len(fitted)} of {len(outcomes)} models.\n")

    print("Applying the models to the full cohort...")
    with_deviations = normative_model.add_all_deviation_scores(fitted, eligible)

    with_deviations.to_csv(args.output, index=False)
    print(f"Wrote {len(with_deviations)} participants to {args.output}")


if __name__ == "__main__":
    main()
