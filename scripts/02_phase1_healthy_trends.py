#!/usr/bin/env python
"""
Step 2 (Phase 1): healthy trends of turbulent dynamics across demographics.

Builds the strictly screened healthy control cohort, compares covariates between
the control and depressive-symptom groups, then fits one linear mixed-effects
model per TD feature to identify which covariates the normative model needs to
account for.

Outputs
-------
    results/phase1_covariate_comparison.csv   group differences in age, sex, motion
    results/phase1_anova_<feature>.csv        omnibus F-tests with partial eta squared
    results/phase1_coefficients_<feature>.csv fixed-effect estimates
    figures/phase1_scale_sensitivity.png/.svg

Usage
-----
    python scripts/02_phase1_healthy_trends.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, plotting  # noqa: E402
from src.cohort import (  # noqa: E402
    apply_common_exclusions,
    compare_covariates,
    load_neurological_exclusions,
    load_psychiatric_exclusions,
    select_depressive_symptoms,
    select_healthy_controls,
)
from src.phase1_lmm import run_phase1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.FEATURES_FILE)
    parser.add_argument("--icd10", type=Path, default=config.ICD10_FILE)
    parser.add_argument(
        "--no-figure", action="store_true", help="Skip figure generation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} participants with TD features.\n")

    neurological = load_neurological_exclusions(args.icd10)
    psychiatric = load_psychiatric_exclusions(args.icd10)

    eligible = apply_common_exclusions(df, neurological)
    controls = select_healthy_controls(eligible, psychiatric)
    cases = select_depressive_symptoms(eligible)

    # Check whether the covariates differ between the groups, which determines
    # what the normative model in Phase 2 must correct for.
    print("\nCovariate comparison, depressive symptoms vs healthy controls:")
    comparison = compare_covariates(cases, controls)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    comparison.to_csv(
        config.RESULTS_DIR / "phase1_covariate_comparison.csv", index=False
    )

    print("\nFitting mixed-effects models on the healthy control cohort...")
    anova_tables, coefficient_tables = run_phase1(controls)

    for name, table in anova_tables.items():
        table.to_csv(config.RESULTS_DIR / f"phase1_anova_{name}.csv")
    for name, table in coefficient_tables.items():
        table.to_csv(config.RESULTS_DIR / f"phase1_coefficients_{name}.csv")

    print(f"\nWrote model tables to {config.RESULTS_DIR}")

    if not args.no_figure:
        print("Generating the Phase 1 figure...")
        plotting.plot_phase1(controls)
        print(f"Wrote figures to {config.FIGURES_DIR}")

    controls.to_csv(config.RESULTS_DIR / "cohort_healthy_controls.csv", index=False)
    cases.to_csv(config.RESULTS_DIR / "cohort_depressive_symptoms.csv", index=False)


if __name__ == "__main__":
    main()
