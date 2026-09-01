"""
Turbulent dynamics (TD) features.

Three features are computed from the parcellated resting-state BOLD signal, each
across a range of spatial scales indexed by the inverse scale parameter lambda:

  Amplitude Turbulence   the variability of the local Kuramoto order parameter
                         over space and time (Text S1.1)
  Information Cascade    the efficiency of coupling between adjacent spatial
  Flow (ICF)             scales, one step apart in time (Text S1.2)
  Information Transfer   the spatial decay of correlations between local order
                         parameters, as the slope of a log-log fit (Text S1.3)

All three are built on the local Kuramoto order parameter under an exponential
distance rule, following Deco & Kringelbach (2020) and Deco et al. (2025).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy import signal

from . import config

# Guards against dividing by an all-but-zero weight sum at very large lambda.
_MIN_DENOMINATOR = 1e-200


# --------------------------------------------------------------------------- #
# Preprocessing of the parcellated signal
# --------------------------------------------------------------------------- #
def make_bandpass_filter(
    tr: float = config.TR,
    low_hz: float = config.BANDPASS_LOW_HZ,
    high_hz: float = config.BANDPASS_HIGH_HZ,
    order: int = config.BANDPASS_ORDER,
):
    """Return the (b, a) coefficients of the Butterworth band-pass filter."""
    nyquist = 0.5 / tr
    return signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")


def bandpass_filter(timeseries: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply the band-pass filter to every parcel with zero phase distortion."""
    return signal.filtfilt(b, a, timeseries, axis=1)


def instantaneous_phase(timeseries: np.ndarray) -> np.ndarray:
    """Return the Hilbert phase of each parcel, shape (n_parcels, n_timepoints)."""
    return np.angle(signal.hilbert(timeseries, axis=1))


def prepare_phases(
    timeseries: np.ndarray, b: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """Band-pass filter the signal and return its instantaneous phase."""
    return instantaneous_phase(bandpass_filter(timeseries, b, a))


# --------------------------------------------------------------------------- #
# Local Kuramoto order parameter
# --------------------------------------------------------------------------- #
def _local_kuramoto_single_scale(
    lam: float, distance_matrix: np.ndarray, phase_exponentials: np.ndarray
):
    """
    Local Kuramoto order parameter modulus at one spatial scale.

    The exponential distance rule W_ij = exp(-lambda * d_ij) defines a weighted
    neighbourhood for every parcel; R_i(t) is the weighted circular mean of the
    phases in that neighbourhood, and its modulus measures local synchronisation.
    """
    weights = np.exp(-lam * distance_matrix)
    np.fill_diagonal(weights, 0.0)  # a parcel does not contribute to its own order

    normaliser = weights.sum(axis=1, keepdims=True)
    normaliser[normaliser < _MIN_DENOMINATOR] = _MIN_DENOMINATOR

    r_complex = (weights @ phase_exponentials) / normaliser
    return lam, np.abs(r_complex)


def local_kuramoto_across_scales(
    phases: np.ndarray,
    distance_matrix: np.ndarray,
    lambda_scales: Sequence[float] = config.LAMBDA_SCALES,
    n_jobs: int = -1,
) -> Dict[float, np.ndarray]:
    """
    Compute the local Kuramoto order parameter modulus at every spatial scale.

    Parameters
    ----------
    phases
        Instantaneous phases, shape (n_parcels, n_timepoints).
    distance_matrix
        Euclidean distances between the same parcels, in millimetres.
    lambda_scales
        Inverse spatial scales to evaluate.
    n_jobs
        Number of workers; scales are independent so they parallelise cleanly.

    Returns
    -------
    Mapping from lambda to an (n_parcels, n_timepoints) array of moduli.
    """
    n_parcels = phases.shape[0]
    if distance_matrix.shape != (n_parcels, n_parcels):
        raise ValueError("Phase and distance matrices describe different parcel sets.")

    # exp(i * phi) is shared across all scales, so compute it once.
    phase_exponentials = np.exp(1j * phases)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_local_kuramoto_single_scale)(
            lam, distance_matrix, phase_exponentials
        )
        for lam in lambda_scales
    )
    return dict(results)


# --------------------------------------------------------------------------- #
# S1.1 Amplitude turbulence
# --------------------------------------------------------------------------- #
def amplitude_turbulence(moduli: np.ndarray) -> float:
    """
    Amplitude turbulence: the standard deviation of the local order parameter
    across both space and time. High values mean synchronisation is unevenly
    distributed, i.e. the dynamics are turbulent rather than uniformly ordered.
    """
    return float(np.std(moduli))


# --------------------------------------------------------------------------- #
# S1.2 Information cascade flow
# --------------------------------------------------------------------------- #
def information_cascade_flow(
    moduli_per_lambda: Dict[float, np.ndarray],
    lambda_scales: Sequence[float] = config.LAMBDA_SCALES,
    lag: int = config.CASCADE_FLOW_LAG,
) -> Dict[float, float]:
    """
    Information cascade flow between adjacent spatial scales.

    For each pair of neighbouring scales, correlate the finer scale's order
    parameter at time t + lag against the adjacent coarser scale's at time t,
    separately per parcel, then average over parcels. This measures how
    efficiently information propagates from coarse to fine spatial scales.

    Returns
    -------
    Mapping from the finer (target) lambda of each pair to its flow value.
    Contains one fewer entry than `lambda_scales`.
    """
    ordered = np.sort(np.unique(np.asarray(lambda_scales, dtype=float)))
    if ordered.size < 2:
        return {}

    flows: Dict[float, float] = {}

    for i in range(1, ordered.size):
        coarse, fine = ordered[i - 1], ordered[i]

        target_future = moduli_per_lambda[fine][:, lag:]
        source_past = moduli_per_lambda[coarse][:, : -lag or None]

        # Per-parcel Pearson correlation over time, computed directly so that
        # constant parcels can be excluded rather than raising.
        target_centred = target_future - target_future.mean(axis=1, keepdims=True)
        source_centred = source_past - source_past.mean(axis=1, keepdims=True)

        numerator = np.sum(target_centred * source_centred, axis=1)
        denominator = np.sqrt(np.sum(target_centred**2, axis=1)) * np.sqrt(
            np.sum(source_centred**2, axis=1)
        )

        correlations = np.full(numerator.shape, np.nan)
        np.divide(numerator, denominator, out=correlations, where=denominator != 0)

        flows[float(fine)] = float(np.nanmean(correlations))

    return flows


# --------------------------------------------------------------------------- #
# S1.3 Information transfer
# --------------------------------------------------------------------------- #
def information_transfer(
    moduli: np.ndarray,
    distance_matrix: np.ndarray,
    n_bins: int = config.TRANSFER_N_BINS,
    first_bin: int = config.TRANSFER_FIT_FIRST_BIN,
    last_bin: int = config.TRANSFER_FIT_LAST_BIN,
) -> float:
    """
    Information transfer: the spatial decay rate of order-parameter correlations.

    Correlations between the order-parameter time series of every parcel pair are
    averaged within Euclidean distance bins, and a line is fitted to the log-log
    relationship between mean correlation and distance over an intermediate range
    of bins. The absolute slope is reported: a steeper slope means correlations
    fall off faster with distance, i.e. less long-range information transfer.

    Returns NaN when fewer than two usable bins fall inside the fitting window.
    """
    correlation_matrix = np.corrcoef(moduli)
    np.fill_diagonal(correlation_matrix, 0.0)

    max_distance = float(np.max(distance_matrix))
    bin_width = max_distance / n_bins
    bin_centres = np.linspace(bin_width / 2, max_distance - bin_width / 2, n_bins)

    upper = np.triu_indices(distance_matrix.shape[0], k=1)
    distances = distance_matrix[upper]
    correlations = correlation_matrix[upper]

    bin_of_pair = np.clip((distances / bin_width).astype(int), 0, n_bins - 1)

    correlation_sums = np.zeros(n_bins)
    pair_counts = np.zeros(n_bins)
    np.add.at(correlation_sums, bin_of_pair, correlations)
    np.add.at(pair_counts, bin_of_pair, 1)

    mean_correlation = np.divide(
        correlation_sums,
        pair_counts,
        out=np.full(n_bins, np.nan),
        where=pair_counts != 0,
    )

    window = np.arange(first_bin, min(last_bin, n_bins))
    usable = window[
        ~np.isnan(mean_correlation[window]) & (mean_correlation[window] > 0)
    ]
    if usable.size < 2:
        return float("nan")

    x = np.log(bin_centres[usable])
    # Normalise by the first usable bin so the intercept is not scale-dependent.
    y = np.log(mean_correlation[usable] / mean_correlation[usable[0]])

    try:
        slope, _ = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return float("nan")

    return float(abs(slope))


# --------------------------------------------------------------------------- #
# Per-subject feature extraction
# --------------------------------------------------------------------------- #
def compute_subject_features(
    timeseries: np.ndarray,
    distance_matrix: np.ndarray,
    network_index: Dict[str, Sequence[int]] | None = None,
    lambda_scales: Sequence[float] = config.LAMBDA_SCALES,
    filter_coefficients: tuple | None = None,
    n_jobs: int = -1,
) -> Dict[str, float]:
    """
    Compute all TD features for one participant.

    Expects the raw parcellated time series and the full distance matrix; parcel
    quality control, filtering, and the Hilbert transform are applied here.

    Returns
    -------
    Flat dictionary of feature names to values, ready to become one row of the
    feature table. Column names follow `<feature>_<lambda>` for whole-brain
    values and `<feature>_<network>_<lambda>` for network-restricted values.
    """
    from .parcellation import filter_rois, remap_network_index

    if filter_coefficients is None:
        filter_coefficients = make_bandpass_filter()
    b, a = filter_coefficients

    timeseries, distance_matrix, kept = filter_rois(timeseries, distance_matrix)
    phases = prepare_phases(timeseries, b, a)

    moduli_per_lambda = local_kuramoto_across_scales(
        phases, distance_matrix, lambda_scales, n_jobs=n_jobs
    )

    networks = (
        remap_network_index(dict(network_index), kept) if network_index else {}
    )

    features: Dict[str, float] = {"n_parcels_kept": int(kept.size)}
    turbulence_values, transfer_values = [], []

    for lam in sorted(moduli_per_lambda):
        moduli = moduli_per_lambda[lam]

        turbulence = amplitude_turbulence(moduli)
        transfer = information_transfer(moduli, distance_matrix)
        turbulence_values.append(turbulence)
        transfer_values.append(transfer)

        features[f"amp_turb_{lam:.3f}"] = turbulence
        features[f"info_transfer_{lam:.3f}"] = transfer

        for name, indices in networks.items():
            if not indices:
                features[f"amp_turb_{name}_{lam:.3f}"] = np.nan
                features[f"info_transfer_{name}_{lam:.3f}"] = np.nan
                continue

            network_moduli = moduli[indices, :]
            network_distances = distance_matrix[np.ix_(indices, indices)]
            features[f"amp_turb_{name}_{lam:.3f}"] = amplitude_turbulence(
                network_moduli
            )
            features[f"info_transfer_{name}_{lam:.3f}"] = information_transfer(
                network_moduli, network_distances
            )

    features["amp_turb_mean"] = float(np.nanmean(turbulence_values))
    features["info_transfer_mean"] = float(np.nanmean(transfer_values))

    flows = information_cascade_flow(moduli_per_lambda, lambda_scales)
    for lam, value in flows.items():
        features[f"info_cascade_flow_{lam:.3f}"] = value
    features["info_cascade_flow_mean"] = (
        float(np.nanmean(list(flows.values()))) if flows else np.nan
    )

    return features
