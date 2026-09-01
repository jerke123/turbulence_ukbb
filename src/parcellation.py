"""
Parcellation geometry.

The turbulence metrics are all defined on the Euclidean distance between parcel
centroids, so this module builds a single 1054 x 1054 distance matrix from the
Schaefer 2018 cortical parcellation (1000 parcels, 7 networks) stacked on top of
the Tian Scale IV subcortical parcellation (54 parcels), in that order.

The same ordering is assumed for the concatenated BOLD time series.
"""

from __future__ import annotations

import io
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import pdist, squareform

from . import config


def fetch_cortical_centroids() -> np.ndarray:
    """Return the (1000, 3) MNI centroid coordinates of the Schaefer parcels."""
    response = requests.get(config.SCHAEFER_CENTROID_URL, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    # R, A, S are the x, y, z coordinates in millimetres.
    return df[["R", "A", "S"]].to_numpy(dtype=float)


def fetch_subcortical_centroids() -> np.ndarray:
    """Return the (54, 3) MNI centre-of-gravity coordinates of the Tian parcels."""
    response = requests.get(config.TIAN_CENTROID_URL, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(
        io.StringIO(response.content.decode("utf-8")), sep=r"\s+", header=None
    )
    if len(df) != config.N_SUBCORTICAL_PARCELS:
        raise ValueError(
            f"Expected {config.N_SUBCORTICAL_PARCELS} subcortical parcels, "
            f"found {len(df)}."
        )
    return df.to_numpy(dtype=float)[:, :3]


def build_distance_matrix() -> np.ndarray:
    """
    Build the full parcel-to-parcel Euclidean distance matrix in millimetres.

    Cortical parcels occupy indices 0..999 and subcortical parcels 1000..1053,
    matching the order in which the two time-series files are concatenated.
    """
    coords = np.vstack([fetch_cortical_centroids(), fetch_subcortical_centroids()])
    return squareform(pdist(coords, metric="euclidean"))


def build_network_index(labels: Sequence[str]) -> Dict[str, List[int]]:
    """
    Map each Yeo network name to the parcel indices belonging to it.

    `labels` are the Schaefer parcel labels (without the background label).
    The subcortex is added as an eighth "network" covering the Tian parcels.
    """
    networks = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    index: Dict[str, List[int]] = {name: [] for name in networks}

    for i, label in enumerate(labels):
        label_str = label.decode() if isinstance(label, bytes) else str(label)
        for name in networks:
            if name in label_str:
                index[name].append(i)

    n_cortical = len(labels)
    index["Subcortex"] = [
        n_cortical + i for i in range(config.N_SUBCORTICAL_PARCELS)
    ]
    return index


def load_network_index() -> Dict[str, List[int]]:
    """Fetch the Schaefer labels via nilearn and build the network index."""
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(
        n_rois=config.N_CORTICAL_PARCELS,
        yeo_networks=config.YEO_NETWORKS,
        resolution_mm=2,
    )
    # The first entry is the background label.
    return build_network_index(atlas["labels"][1:])


def filter_rois(
    timeseries: np.ndarray,
    distance_matrix: np.ndarray,
    std_threshold: float = config.ROI_STD_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop parcels with unusable time series and subset the distance matrix to match.

    A parcel is dropped if its time series contains any NaN, is entirely zero, or
    has a standard deviation below `std_threshold` (a flat signal carries no
    phase information and would distort the local order parameter).

    Parameters
    ----------
    timeseries
        BOLD signal of shape (n_parcels, n_timepoints).
    distance_matrix
        Square distance matrix of shape (n_parcels, n_parcels).
    std_threshold
        Minimum standard deviation for a parcel to be retained.

    Returns
    -------
    filtered_timeseries
        Shape (n_kept, n_timepoints).
    filtered_distance_matrix
        Shape (n_kept, n_kept).
    kept
        Sorted array of the original indices that were retained, needed to map
        network definitions onto the filtered data.
    """
    n_parcels = timeseries.shape[0]
    if distance_matrix.shape != (n_parcels, n_parcels):
        raise ValueError(
            f"Time series has {n_parcels} parcels but the distance matrix is "
            f"{distance_matrix.shape}; the two must agree."
        )

    finite = ~np.isnan(timeseries).any(axis=1)
    non_zero = ~np.all(timeseries == 0, axis=1)

    # Only evaluate variance on parcels that are already known to be finite, so
    # that all-NaN rows do not raise an empty-slice warning.
    variable = np.zeros(n_parcels, dtype=bool)
    variable[finite] = timeseries[finite].std(axis=1) >= std_threshold

    keep = finite & non_zero & variable

    if not keep.any():
        raise ValueError("No parcels survived quality control for this subject.")

    kept = np.flatnonzero(keep)
    return timeseries[keep, :], distance_matrix[np.ix_(kept, kept)], kept


def remap_network_index(
    network_index: Dict[str, List[int]], kept: np.ndarray
) -> Dict[str, List[int]]:
    """
    Translate network definitions from original parcel indices to filtered ones.

    Networks whose parcels were all dropped map to an empty list.
    """
    lookup = {original: new for new, original in enumerate(kept)}
    return {
        name: [lookup[i] for i in indices if i in lookup]
        for name, indices in network_index.items()
    }
