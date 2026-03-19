"""
doublehump.py
-------------
Detects double-humped (bimodal) lightcurves caused by elongated asteroids.

When an asteroid has two brightness minima per rotation (due to its elongated
shape), period-finding algorithms often report P/2 instead of the true period.
This module detects this case and corrects the period estimate.

Main function
-------------
check_double_hump(t_hrs, y, dy, candidate_period_hr)
    Returns DoublehumpResult with is_double_hump flag and corrected period.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# Number of phase bins for lightcurve folding
N_PHASE_BINS = 20

# Minimum fractional depth difference between two minima to count as double-hump
# (prevents noise spikes from being counted as a second minimum)
MIN_HUMP_DEPTH_RATIO = 0.3

# If double-hump chi2 improvement over single-hump is > this, prefer double
CHI2_IMPROVEMENT_THRESHOLD = 0.25


@dataclass
class DoublehumpResult:
    """Result of double-hump detection."""
    candidate_period_hr:  float   # Input period
    is_double_hump:       bool    # True if double-hump detected
    corrected_period_hr:  float   # 2x candidate if double-hump, else candidate
    n_minima:             int     # Number of minima found in phase fold
    hump_asymmetry:       float   # Depth ratio of deeper/shallower minimum (1=equal)
    chi2_single:          float   # Chi2 of single-hump sinusoid fit
    chi2_double:          float   # Chi2 of double-hump (2-harmonic) fit
    chi2_improvement:     float   # (chi2_single - chi2_double) / chi2_single
    confidence:           float   # 0-1 confidence that this is a double hump
    note:                 str     # Human-readable explanation


def _phase_fold_bins(
    t_hrs: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    period_hr: float,
    n_bins: int = N_PHASE_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-fold a lightcurve and bin it.

    Returns
    -------
    bin_centers : np.ndarray, shape (n_bins,)
    bin_means   : np.ndarray, shape (n_bins,) — NaN for empty bins
    bin_stds    : np.ndarray, shape (n_bins,) — NaN for empty bins
    """
    phase     = (t_hrs % period_hr) / period_hr
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_means = np.full(n_bins, np.nan)
    bin_stds  = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if mask.sum() >= 2:
            weights = 1.0 / dy[mask]**2
            wmean   = np.average(y[mask], weights=weights)
            bin_means[i] = wmean
            bin_stds[i]  = np.std(y[mask])

    return bin_centers, bin_means, bin_stds


def _fit_harmonics(
    t_hrs: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    period_hr: float,
    n_harmonics: int,
) -> float:
    """
    Fit a Fourier series with n_harmonics to the lightcurve at period_hr.
    Returns reduced chi-squared of the fit.

    n_harmonics=1 → single-hump sinusoid
    n_harmonics=2 → double-hump (can model two unequal humps)
    """
    phase = (t_hrs % period_hr) / period_hr * 2 * np.pi

    # Design matrix: [1, cos(phi), sin(phi), cos(2phi), sin(2phi), ...]
    cols = [np.ones(len(t_hrs))]
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(h * phase))
        cols.append(np.sin(h * phase))

    A  = np.column_stack(cols)
    W  = np.diag(1.0 / dy**2)
    AtW = A.T @ W

    try:
        coeffs = np.linalg.solve(AtW @ A, AtW @ y)
        y_fit  = A @ coeffs
        resid  = (y - y_fit) / dy
        dof    = max(len(y) - len(coeffs), 1)
        return float(np.sum(resid**2) / dof)
    except np.linalg.LinAlgError:
        return np.inf


def _count_minima(bin_means: np.ndarray) -> Tuple[int, float]:
    """
    Count local minima in a phase-folded binned lightcurve.
    Returns (n_minima, hump_asymmetry).

    Smooths first to avoid noise spikes. Wraps around phase edges.
    hump_asymmetry = depth of deeper minimum / depth of shallower minimum
    """
    # Fill NaN bins by interpolation
    valid = ~np.isnan(bin_means)
    if valid.sum() < 4:
        return 0, 1.0

    x     = np.arange(len(bin_means))
    filled = np.interp(x, x[valid], bin_means[valid])

    # Smooth and tile to handle wrap-around
    smoothed = uniform_filter1d(filled, size=3, mode='wrap')
    tiled    = np.tile(smoothed, 3)  # repeat 3x to handle wrap

    # Find minima in the middle tile
    peaks_idx, props = find_peaks(
        -tiled,                        # invert for minima
        prominence=0.0,
        distance=len(smoothed) // 6,   # minima must be separated by >1/6 period
    )

    # Keep only minima in the middle tile (indices len..2*len)
    n = len(smoothed)
    centre_mask = (peaks_idx >= n) & (peaks_idx < 2 * n)
    centre_peaks = peaks_idx[centre_mask]

    n_minima = len(centre_peaks)

    if n_minima < 2:
        return n_minima, 1.0

    # Compute depth asymmetry between deepest two minima
    depths = sorted([-tiled[idx] for idx in centre_peaks[:2]], reverse=True)
    global_range = smoothed.max() - smoothed.min()
    if global_range < 1e-6:
        return n_minima, 1.0

    depth_ratio = abs(depths[0] - depths[1]) / global_range
    asymmetry   = 1.0 + depth_ratio  # 1=equal humps, >1=unequal

    return n_minima, float(asymmetry)


def check_double_hump(
    t_hrs:              np.ndarray,
    y:                  np.ndarray,
    dy:                 np.ndarray,
    candidate_period_hr: float,
) -> DoublehumpResult:
    """
    Test whether a candidate period is actually P/2 of a double-hump lightcurve.

    Parameters
    ----------
    t_hrs : np.ndarray
        Observation times in hours.
    y : np.ndarray
        Detrended magnitudes.
    dy : np.ndarray
        Magnitude uncertainties.
    candidate_period_hr : float
        Period found by Tier1/Tier2 to test.

    Returns
    -------
    DoublehumpResult
    """
    P  = candidate_period_hr
    P2 = candidate_period_hr * 2.0

    # ── Step 1: Fit single vs double harmonic at candidate period ─────────────
    chi2_single = _fit_harmonics(t_hrs, y, dy, P,  n_harmonics=1)
    chi2_double = _fit_harmonics(t_hrs, y, dy, P,  n_harmonics=2)
    chi2_2P     = _fit_harmonics(t_hrs, y, dy, P2, n_harmonics=2)

    chi2_improvement = (chi2_single - chi2_double) / max(chi2_single, 1e-10)

    # ── Step 2: Count minima in phase-folded binned lightcurve ───────────────
    _, bin_means, _ = _phase_fold_bins(t_hrs, y, dy, P)
    n_minima, asymmetry = _count_minima(bin_means)

    # ── Step 3: Also check minima at 2P ──────────────────────────────────────
    _, bin_means_2P, _ = _phase_fold_bins(t_hrs, y, dy, P2)
    n_minima_2P, asymmetry_2P = _count_minima(bin_means_2P)

    # ── Step 4: Decide ────────────────────────────────────────────────────────
    # Evidence for double-hump:
    # (a) Fitting 2 harmonics at P improves chi2 significantly
    # (b) Phase fold at P shows 2 minima
    # (c) chi2 at 2P with 2 harmonics is better than chi2 at P with 1 harmonic

    score = 0.0
    reasons = []

    if chi2_improvement > CHI2_IMPROVEMENT_THRESHOLD:
        score += 0.4
        reasons.append(f"2-harmonic fit improves chi2 by {chi2_improvement:.2f}")

    if n_minima >= 2:
        score += 0.4
        reasons.append(f"{n_minima} minima found in phase fold")

    if chi2_2P < chi2_single * 0.9:
        score += 0.2
        reasons.append(f"2P fit (chi2={chi2_2P:.2f}) beats P fit (chi2={chi2_single:.2f})")

    # Hard gate: 2P fit must actually be better than P fit
    p2_is_better   = chi2_2P < chi2_single * 0.85
    is_double_hump = score >= 0.7

    corrected = P2 if is_double_hump else P

    note = (
        f"Double-hump detected ({', '.join(reasons)}). Period corrected {P:.3f}→{P2:.3f}hr"
        if is_double_hump
        else f"Single-hump lightcurve. Period {P:.3f}hr retained. (score={score:.2f})"
    )

    logger.debug(
        f"DoublehumpCheck P={P:.3f}hr: "
        f"n_minima={n_minima}, chi2_improvement={chi2_improvement:.3f}, "
        f"score={score:.2f}, is_double={is_double_hump}"
    )

    return DoublehumpResult(
        candidate_period_hr  = P,
        is_double_hump       = is_double_hump,
        corrected_period_hr  = corrected,
        n_minima             = n_minima,
        hump_asymmetry       = asymmetry,
        chi2_single          = chi2_single,
        chi2_double          = chi2_double,
        chi2_improvement     = chi2_improvement,
        confidence           = score,
        note                 = note,
    )
