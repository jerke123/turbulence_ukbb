"""
Loading parcellated resting-state time series.

The cortical and subcortical parcellations are distributed as separate files per
participant. This module concatenates them into a single array in the order the
distance matrix expects (cortical parcels first, then subcortical).

Access to the imaging archives is platform-specific and governed by the data
access agreement, so the retrieval step is kept behind a small interface: point
`SubjectFileLocator` at a directory layout, or supply your own callable that
returns the two file paths for a participant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from . import config


@dataclass
class SubjectFileLocator:
    """
    Resolve a participant id to their two parcellated time-series files.

    Assumes a directory laid out as::

        root/
          <eid>/
            fMRI.Schaefer7n1000p.csv.gz
            fMRI.Tian_Subcortex_S4_3T.csv.gz

    Adapt `resolve` if the local extraction of the imaging archives differs.
    """

    root: Path
    cortical_file: str = config.CORTICAL_TIMESERIES_FILE
    subcortical_file: str = config.SUBCORTICAL_TIMESERIES_FILE

    def resolve(self, subject_id: str) -> Tuple[Path, Path]:
        subject_dir = Path(self.root) / str(subject_id)
        return (
            subject_dir / self.cortical_file,
            subject_dir / self.subcortical_file,
        )

    def __call__(self, subject_id: str) -> Tuple[Path, Path]:
        return self.resolve(subject_id)


def load_timeseries(cortical_path: Path, subcortical_path: Path) -> np.ndarray:
    """
    Read and concatenate one participant's parcellated time series.

    Each file has one row per parcel, a leading label column, and one column per
    volume. Returns an array of shape (n_parcels, n_timepoints) with the
    cortical parcels first.
    """
    cortical = pd.read_csv(cortical_path, compression="gzip")
    subcortical = pd.read_csv(subcortical_path, compression="gzip")

    if cortical.empty or subcortical.empty:
        raise ValueError("Empty time-series file.")

    combined = pd.concat([cortical, subcortical], ignore_index=True)

    # The first column holds the parcel label; everything after it is signal.
    return combined.iloc[:, 1:].to_numpy(dtype=float)


def load_subject(
    subject_id: str, locator: Callable[[str], Tuple[Path, Path]]
) -> np.ndarray:
    """Locate and load one participant's concatenated time series."""
    cortical_path, subcortical_path = locator(subject_id)

    for path in (cortical_path, subcortical_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing time-series file: {path}")

    return load_timeseries(cortical_path, subcortical_path)
