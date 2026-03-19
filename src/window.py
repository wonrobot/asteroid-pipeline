"""
window.py
---------
Spectral window function computation and window-informed peak selection.

The window function reveals which periods are aliases of the sampling cadence.
Peaks in the data periodogram that coincide with window peaks are penalised
during period selection.
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# How strongly to penalise window-contaminated peaks (0=none, 1=full suppression)
WINDOW_PENALTY_ALPHA = 0.7

# A peak is "contaminated" if window power > this fraction of max window power
CONTAMINATION_THRESHOLD = 0.2

# Search radius around a candidate period when checking window contamination
# expressed as fraction of the period
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
