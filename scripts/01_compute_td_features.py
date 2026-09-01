#!/usr/bin/env python
"""
Step 1: compute turbulent dynamics features for every participant.

Reads the phenotype table, derives the cohort variables, then loops over
participants computing amplitude turbulence, information cascade flow, and
information transfer at each spatial scale. Results are written in chunks so a
long run can be resumed after an interruption.

Usage
-----
    python scripts/01_compute_td_features.py --timeseries-root /path/to/timeseries

    # resume a partial run
    python scripts/01_compute_td_features.py --timeseries-root ... --start-chunk 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, parcellation, turbulence  # noqa: E402
from src.cohort import load_phenotypes  # noqa: E402
from src.io_timeseries import SubjectFileLocator, load_subject  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeseries-root",
        type=Path,
        required=True,
        help="Directory containing one subdirectory of parcellated time series "
             "per participant.",
    )
    parser.add_argument(
        "--phenotypes",
        type=Path,
        default=config.PHENOTYPE_FILE,
        help="Phenotype export (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.FEATURES_FILE,
        help="Where to write the merged feature table (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Participants per intermediate output file (default: %(default)s).",
    )
    parser.add_argument(
        "--start-chunk", type=int, default=1,
        help="First chunk to process, 1-indexed (default: %(default)s).",
    )
    parser.add_argument(
        "--end-chunk", type=int, default=None,
        help="Last chunk to process; defaults to all remaining chunks.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Workers used across spatial scales (default: all cores).",
    )
    parser.add_argument(
        "--skip-networks", action="store_true",
        help="Compute whole-brain features only, omitting the per-network "
             "breakdown. Much faster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()

    print("Loading phenotypes and deriving cohort variables...")
    phenotypes = load_phenotypes(args.phenotypes)
    print(f"  {len(phenotypes)} participants with usable phenotype data")

    print("Building the parcel distance matrix...")
    distance_matrix = parcellation.build_distance_matrix()
    print(f"  {distance_matrix.shape[0]} parcels, "
          f"maximum distance {distance_matrix.max():.1f} mm")

    network_index = None
    if not args.skip_networks:
        print("Loading the network assignment of each parcel...")
        network_index = parcellation.load_network_index()

    locator = SubjectFileLocator(root=args.timeseries_root)
    filter_coefficients = turbulence.make_bandpass_filter()

    chunk_dir = config.RESULTS_DIR / "feature_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = max(1, int(np.ceil(len(phenotypes) / args.chunk_size)))
    chunks = np.array_split(phenotypes, n_chunks)
    end_chunk = args.end_chunk or n_chunks

    print(f"\nProcessing chunks {args.start_chunk}-{end_chunk} of {n_chunks}.")

    for chunk_number in range(args.start_chunk, end_chunk + 1):
        chunk = chunks[chunk_number - 1]
        output_path = chunk_dir / f"features_chunk_{chunk_number:03d}.csv"

        if output_path.exists():
            print(f"Chunk {chunk_number}: already done, skipping.")
            continue

        print(f"\nChunk {chunk_number} ({len(chunk)} participants)")
        rows = []
        n_failed = 0

        for _, participant in tqdm(
            chunk.iterrows(), total=len(chunk), desc=f"chunk {chunk_number}"
        ):
            subject_id = participant[config.COL_SUBJECT]
            try:
                timeseries = load_subject(subject_id, locator)
                features = turbulence.compute_subject_features(
                    timeseries,
                    distance_matrix,
                    network_index=network_index,
                    filter_coefficients=filter_coefficients,
                    n_jobs=args.n_jobs,
                )
                rows.append({**participant.to_dict(), **features})
            except Exception as error:
                n_failed += 1
                tqdm.write(f"  {subject_id}: {type(error).__name__}: {error}")

        if rows:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"  wrote {len(rows)} rows to {output_path} ({n_failed} failed)")
        else:
            print(f"  no participants processed successfully ({n_failed} failed)")

    chunk_files = sorted(chunk_dir.glob("features_chunk_*.csv"))
    if chunk_files:
        print(f"\nMerging {len(chunk_files)} chunk files...")
        merged = pd.concat(
            (pd.read_csv(path) for path in chunk_files), ignore_index=True
        )
        merged = merged.dropna(
            subset=[f"amp_turb_{config.LAMBDA_SCALES[0]:.3f}"]
        )
        merged.to_csv(args.output, index=False)
        print(f"Wrote {len(merged)} participants to {args.output}")


if __name__ == "__main__":
    main()
