"""
window.py
---------
Spectral window function computation and window-informed peak selection.

Status: implemented but not yet wired into the main pipeline.

Planned uses
------------
1. Periodogram plots — vertical lines at window-dominant frequencies so
   alias peaks are visually obvious when inspecting individual objects.
2. Alias identification — contamination_score() can be used in Tier 1/2
   to automatically down-weight candidates that fall on window peaks,
   supplementing the existing ALIAS_PERIODS_HR list in reliability.py
   with cadence-specific aliases rather than just fixed daily/annual ones.
3. CLEAN input — the window DFT is already computed inside tier3.clean_periodogram;
   unifying with this module would avoid duplication.

See window_informed_peaks() and best_clean_period() for the main entry points.

The window function reveals which periods are aliases of the sampling cadence.
Peaks in the data periodogram that coincide with window peaks are penalised
during period selection.
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Penalty strength for window-contaminated peaks in window_informed_peaks().
# Adjusted power = data_power * (1 - alpha * contamination_score).
#
# Scientific basis: alpha controls the trade-off between alias rejection and
# signal recovery. alpha=0 is pure data power (no alias awareness); alpha=1
# fully suppresses any period at a window peak. The value 0.7 is used as a
# provisional default pending Change 3 (threshold calibration from recovery
# rate curves on labelled data). It is NOT derived from the Greenstreet+2026
# test set and is NOT a classification threshold — it only affects peak
# ranking within Tier 1, not the pass/fail decision.
# Note: the correct treatment (Change 3) is to derive alpha from simulations
# showing recovery rate vs. contamination score for synthetic lightcurves.
WINDOW_PENALTY_ALPHA = 0.7

# Contamination score above which a period is considered to sit "on" a window
# peak for logging and R-code capping purposes (not a classification veto).
#
# Scientific basis: the contamination score is defined as
#   score = max_window_power_in_radius / max_window_power_global  ∈ [0,1]
# A score of 0.2 means the local window power is ≥20% of the global maximum.
# This is the threshold below which a period is considered to be in a
# window-clean region. The value 0.2 was chosen as a conservative lower bound —
# it only flags periods that are genuinely prominent in the window spectrum,
# not background ripple. The correct derivation (Change 3) is to compare this
# score against the distribution of contamination scores for confirmed real
# periods in a labelled dataset.
CONTAMINATION_THRESHOLD = 0.2

# Fractional search radius for contamination_score().
# When checking whether a candidate period P sits on a window peak, we look
# for the maximum window power within ±(CONTAMINATION_RADIUS × P) of P.
#
# Scientific basis: this must be large enough to capture the main lobe of a
# window function peak without capturing adjacent peaks. For a dataset with
# baseline T days, window peaks have a characteristic width of ~1/T in
# frequency, which translates to ~P²/T in period at period P. For typical
# Rubin commissioning data (T≈12 days, P≈5hr): width ≈ 25/12 ≈ 2hr, so
# CONTAMINATION_RADIUS ≈ 2/5 ≈ 0.4 would be needed. The value 0.03 (3%)
# is therefore UNDER-ESTIMATING the peak width for short baselines — this is
# an outstanding issue flagged for Change 3.
CONTAMINATION_RADIUS = 0.03


def compute_window_function(
    t_hrs: np.ndarray,
    periods_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the spectral window function for a given time sampling.

    The window function is the GLS power spectrum of a constant signal
    (all ones) at the observation times. Any peak in the window function
    represents a period that the sampling cadence can introduce as an alias.

    Parameters
    ----------
    t_hrs : np.ndarray
        Observation times in hours.
    periods_grid : np.ndarray
        Period grid in hours to evaluate window on.

    Returns
    -------
    window_power : np.ndarray, same shape as periods_grid
    """
    from astropy.timeseries import LombScargle

    y_ones  = np.ones(len(t_hrs))
    dy_ones = np.ones(len(t_hrs)) * 0.001
    freqs   = 1.0 / periods_grid

    ls = LombScargle(t_hrs, y_ones, dy_ones, fit_mean=False, center_data=False)
    return ls.power(freqs)


def contamination_score(
    period_hr: float,
    periods_grid: np.ndarray,
    window_power: np.ndarray,
) -> float:
    """
    Return the window contamination score for a candidate period.

    Score = max window power within ±CONTAMINATION_RADIUS of the period,
    normalised to max window power overall.

    Score = 0.0 → period is in a window-clean region (trustworthy)
    Score = 1.0 → period sits exactly on a window peak (likely alias)

    Parameters
    ----------
    period_hr : float
        Candidate period in hours.
    periods_grid : np.ndarray
        Period grid used for window computation.
    window_power : np.ndarray
        Window function power on periods_grid.

    Returns
    -------
    float in [0, 1]
    """
    radius  = CONTAMINATION_RADIUS * period_hr
    mask    = np.abs(periods_grid - period_hr) <= radius
    if not mask.any():
        return 0.0

    local_max  = window_power[mask].max()
    global_max = window_power.max()
    if global_max < 1e-10:
        return 0.0

    return float(np.clip(local_max / global_max, 0.0, 1.0))


def window_informed_peaks(
    periods_grid: np.ndarray,
    data_power: np.ndarray,
    window_power: np.ndarray,
    n_peaks: int = 5,
    min_period_hr: float = 0.5,
    max_period_hr: float = 24.0,
    alpha: float = WINDOW_PENALTY_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the top N peaks in the data periodogram, penalised for window
    contamination.

    Adjusted power = data_power * (1 - alpha * contamination_score)

    Parameters
    ----------
    periods_grid : np.ndarray
    data_power : np.ndarray
    window_power : np.ndarray
    n_peaks : int
        Number of top peaks to return.
    min_period_hr, max_period_hr : float
        Period range to search.
    alpha : float
        Penalty strength. 0 = no penalty, 1 = full suppression of aliases.

    Returns
    -------
    peak_periods : np.ndarray, shape (n_peaks,)
        Periods of top peaks, sorted by adjusted power descending.
    peak_raw_power : np.ndarray, shape (n_peaks,)
        Raw (unpenalised) GLS power at each peak.
    peak_contamination : np.ndarray, shape (n_peaks,)
        Contamination scores for each peak (0=clean, 1=alias).
    """
    # Restrict to valid period range
    mask = (periods_grid >= min_period_hr) & (periods_grid <= max_period_hr)
    p    = periods_grid[mask]
    dp   = data_power[mask]
    wp   = window_power[mask]

    # Compute per-point contamination and adjusted power
    global_max_wp = window_power.max()
    if global_max_wp > 1e-10:
        cont = np.clip(wp / global_max_wp, 0.0, 1.0)
    else:
        cont = np.zeros_like(wp)

    adj_power = dp * (1.0 - alpha * cont)

    # Find local maxima (simple: point higher than both neighbours)
    peak_idx = []
    for i in range(1, len(adj_power) - 1):
        if adj_power[i] > adj_power[i-1] and adj_power[i] > adj_power[i+1]:
            peak_idx.append(i)

    if len(peak_idx) == 0:
        peak_idx = [np.argmax(adj_power)]

    peak_idx = np.array(peak_idx)

    # Sort by adjusted power descending
    order = np.argsort(adj_power[peak_idx])[::-1]
    peak_idx = peak_idx[order[:n_peaks]]

    peak_periods       = p[peak_idx]
    peak_raw_power     = dp[peak_idx]
    peak_contamination = np.array([
        contamination_score(per, periods_grid, window_power)
        for per in peak_periods
    ])

    logger.debug(
        f"Window-informed peaks: "
        + ", ".join(
            f"{per:.3f}hr (cont={c:.2f})"
            for per, c in zip(peak_periods, peak_contamination)
        )
    )

    return peak_periods, peak_raw_power, peak_contamination


def best_clean_period(
    periods_grid: np.ndarray,
    data_power: np.ndarray,
    window_power: np.ndarray,
    alpha: float = WINDOW_PENALTY_ALPHA,
    min_period_hr: float = 0.5,
    max_period_hr: float = 24.0,
) -> Tuple[float, float, float]:
    """
    Return the single best window-cleaned period estimate.

    Parameters
    ----------
    (same as window_informed_peaks)

    Returns
    -------
    period_hr : float
        Best period after window penalisation.
    raw_power : float
        Raw GLS power at this period.
    contamination : float
        Contamination score (0=clean, 1=alias).
    """
    peak_periods, peak_raw_power, peak_cont = window_informed_peaks(
        periods_grid, data_power, window_power,
        n_peaks=10, alpha=alpha,
        min_period_hr=min_period_hr,
        max_period_hr=max_period_hr,
    )

    # Prefer the top peak. If it is heavily contaminated, warn.
    best_period = float(peak_periods[0])
    best_power  = float(peak_raw_power[0])
    best_cont   = float(peak_cont[0])

    if best_cont > CONTAMINATION_THRESHOLD:
        logger.warning(
            f"Best period {best_period:.3f}hr has contamination={best_cont:.2f} "
            f"— likely a sampling alias. Consider 2x period = {best_period*2:.3f}hr"
        )

    return best_period, best_power, best_cont
