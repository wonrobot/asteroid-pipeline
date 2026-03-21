"""
tier1.py
--------
Tier 1: Fast screening of all asteroids.
Runs GLS + MBLS Nterms=1 on every object that passes quality cuts.

Goal: eliminate objects with no detectable periodicity quickly.
Cost: ~milliseconds per asteroid (parallelisable).

Data routing
------------
GLS  → data.y_dt        (merged, band-offset + detrended)
MBLS → data.y_multiband (geometry-corrected only, raw band labels)
       MBLS fits per-band means internally — feeding it pre-offset
       data discards the inter-band colour info it exploits.

Window function
---------------
The spectral window function is computed once here using the actual
observation timestamps. It reveals which periods are aliases of the
sampling cadence (as opposed to the fixed daily/annual alias list).
The window_power array and per-period contamination scores are stored
in Tier1Result so Tier 2 and reliability.py can reuse them without
recomputing.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from gatspy.periodic import LombScargleMultiband

from config import PipelineConfig, DEFAULT_CONFIG
from preprocessing import PreparedData
from window import (
    compute_window_function,
    contamination_score,
    window_informed_peaks,
    CONTAMINATION_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Tier1Result:
    """
    Output of Tier 1 screening.

    Attributes
    ----------
    provid              : asteroid designation
    passes              : True if this object should proceed to Tier 2
    best_period_gls     : best period from GLS after window penalisation (hours)
    best_period_mbls    : best period from MBLS Nterms=1 (hours)
    gls_power_max       : peak GLS power (0–1), raw (pre-penalisation)
    snr                 : amplitude SNR from preprocessing
    n_obs               : number of observations
    reject_reason       : if passes=False, explains why (else None)
    test_periods        : period grid used
    gls_power           : full GLS power array (for plotting)
    mbls_power          : full MBLS power array (for plotting)
    window_power        : spectral window function on test_periods grid.
                          Peaks here are cadence aliases — used downstream
                          by Tier 2 and reliability.py to score candidates.
    gls_contamination   : alias contamination score for best GLS period [0–1].
                          0 = clean, 1 = sits exactly on a window peak.
    mbls_contamination  : alias contamination score for best MBLS period [0–1].
    """
    provid:              str
    passes:              bool
    best_period_gls:     float
    best_period_mbls:    float
    gls_power_max:       float
    snr:                 float
    n_obs:               int
    reject_reason:       Optional[str]
    test_periods:        np.ndarray
    gls_power:           np.ndarray
    mbls_power:          np.ndarray
    # ── New: window function fields ───────────────────────────────────────────
    window_power:        np.ndarray   # window function on same grid as test_periods
    gls_contamination:   float        # cadence-alias score for best GLS period
    mbls_contamination:  float        # cadence-alias score for best MBLS period


# ── Main Tier 1 entry point ───────────────────────────────────────────────────

def run_tier1(
    data:   PreparedData,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tier1Result:
    """
    Run Tier 1 screening on a single prepared asteroid.

    Checks:
    1. Minimum observation count
    2. Minimum SNR
    3. GLS peak power > 0.05 (weak periodicity filter)
    """
    cfg_t = config.tier
    cfg_p = config.period

    # ── Hard quality gates ────────────────────────────────────────────────────
    if data.n_obs < cfg_t.min_obs:
        return _reject(data, config,
                       f"n_obs={data.n_obs} < min_obs={cfg_t.min_obs}")

    if data.snr < cfg_t.snr_threshold:
        return _reject(data, config,
                       f"SNR={data.snr:.2f} < threshold={cfg_t.snr_threshold}")

    # ── Period grid ───────────────────────────────────────────────────────────
    # Tier 1 grid — two-pass strategy with window-qualified expansion
    #
    # Pass 1: coarse search from 0.5hr floor (2000 pts, always fast ~1s).
    #         Window function computed on this coarse grid immediately after
    #         so we can qualify the Pass 2 trigger.
    #
    # Pass 2: expand to Nyquist floor only when BOTH conditions hold:
    #   (a) power is weak OR best period is near the 0.5hr boundary, AND
    #   (b) the coarse best period is NOT window-contaminated
    #
    # Rationale for condition (b):
    #   Low coarse power has two explanations:
    #     i.  True period is below 0.5hr (fast rotator) — expansion helps.
    #     ii. Coarse best period is a cadence alias; real signal is elsewhere
    #         on the same grid — expansion wastes time and may find a spurious
    #         sub-0.5hr alias peak instead.
    #   We now distinguish these cases using the window function.
    #   If the coarse best is contaminated, we keep the coarse grid and let
    #   window-penalised peak selection (below) find the real signal.
    #   This avoids unnecessary 20k-point grid expansions and prevents
    #   cadence aliases from triggering fast-rotator searches.
    p_max_t1    = min(cfg_p.period_max_hr, data.baseline_hr)
    FAST_THRESH = 0.5  # hr

    # Pass 1: coarse grid from 0.5hr (fast for all objects)
    p_min_coarse   = max(data.period_min_hr, FAST_THRESH)
    test_periods   = np.linspace(p_min_coarse, p_max_t1, cfg_p.n_grid_coarse)
    gls_pow_coarse = gls_periodogram(data.t_hrs, data.y_dt, data.dy, test_periods)
    best_coarse    = test_periods[np.argmax(gls_pow_coarse)]
    max_pow_coarse = float(gls_pow_coarse.max())

    # Window function on coarse grid — needed to qualify Pass 2 trigger
    window_pow_coarse = compute_window_function(data.t_hrs, test_periods)
    coarse_cont       = contamination_score(best_coarse, test_periods, window_pow_coarse)
    coarse_contaminated = coarse_cont > CONTAMINATION_THRESHOLD

    near_boundary   = best_coarse < FAST_THRESH * 1.5
    weak_power      = max_pow_coarse < 0.15
    can_search_fast = data.period_min_hr < FAST_THRESH

    # Pass 2 trigger: power condition fires AND coarse best is not a known alias
    # near_boundary always expands — a period at 0.6hr could be 0.5hr P/2;
    # this is a genuine harmonic concern unrelated to alias contamination.
    should_expand = (near_boundary or (weak_power and not coarse_contaminated)) and can_search_fast

    if coarse_contaminated and weak_power and not near_boundary:
        logger.debug(
            f"{data.provid}: Pass 2 suppressed — coarse best={best_coarse:.3f}hr "
            f"is window-contaminated (cont={coarse_cont:.2f}); "
            f"low power is likely alias suppression, not a fast rotator"
        )

    if should_expand:
        n_fast       = min(20_000, max(cfg_p.n_grid_coarse,
                           int(5 * data.baseline_hr
                               * (1.0/data.period_min_hr - 1.0/p_max_t1))))
        test_periods = np.linspace(data.period_min_hr, p_max_t1, n_fast)
        logger.debug(
            f"{data.provid}: Pass 2 expanded grid to {len(test_periods)} pts "
            f"(near_boundary={near_boundary}, weak={weak_power}, "
            f"coarse_cont={coarse_cont:.2f})"
        )
    # else: keep coarse test_periods — normal rotator or contaminated alias

    # ── GLS — uses merged, band-offset-corrected + detrended series ───────────
    gls_pow  = gls_periodogram(data.t_hrs, data.y_dt, data.dy, test_periods)
    gls_max  = float(gls_pow.max())

    # ── Window function on final grid ─────────────────────────────────────────
    # If the grid was not expanded, test_periods == coarse grid and we already
    # have window_pow_coarse — reuse it to avoid a redundant computation.
    # If Pass 2 expanded the grid, recompute on the new (larger) grid.
    if should_expand:
        window_pow = compute_window_function(data.t_hrs, test_periods)
    else:
        window_pow = window_pow_coarse  # same grid, reuse

    # ── Window-penalised best GLS period ─────────────────────────────────────
    # Instead of raw argmax, use window_informed_peaks which applies a
    # penalty to candidates sitting on window peaks.
    # If the top penalised peak differs from raw argmax, the raw best period
    # was a cadence alias and we've caught it here at negligible extra cost.
    gls_peaks, gls_raw_powers, _ = window_informed_peaks(
        test_periods, gls_pow, window_pow,
        n_peaks=3,
        min_period_hr=float(test_periods[0]),
        max_period_hr=float(test_periods[-1]),
    )
    best_gls     = float(gls_peaks[0]) if len(gls_peaks) > 0 else test_periods[np.argmax(gls_pow)]
    gls_raw_best = float(test_periods[np.argmax(gls_pow)])
    gls_cont     = contamination_score(best_gls, test_periods, window_pow)

    if best_gls != gls_raw_best:
        logger.debug(
            f"{data.provid}: GLS raw best={gls_raw_best:.3f}hr is window-contaminated "
            f"→ window-penalised best={best_gls:.3f}hr"
        )
    elif gls_cont > CONTAMINATION_THRESHOLD:
        logger.debug(
            f"{data.provid}: GLS best={best_gls:.3f}hr is window-contaminated "
            f"(score={gls_cont:.2f}) — cadence alias risk"
        )

    # ── MBLS Nterms=1 — uses raw multiband series, NO pre-applied offsets ─────
    try:
        mbls_pow  = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods, nterms=config.period.mbls_nterms_t1
        )
        best_mbls = test_periods[np.argmax(mbls_pow)]
    except Exception as e:
        logger.warning(f"{data.provid}: MBLS Tier1 failed ({e}) — using GLS only")
        mbls_pow  = gls_pow.copy()
        best_mbls = best_gls

    # Score MBLS best period for window contamination
    mbls_cont = contamination_score(float(best_mbls), test_periods, window_pow)

    if mbls_cont > CONTAMINATION_THRESHOLD:
        logger.debug(
            f"{data.provid}: MBLS best={best_mbls:.3f}hr is window-contaminated "
            f"(score={mbls_cont:.2f})"
        )

    # ── Summary log ──────────────────────────────────────────────────────────
    logger.debug(
        f"{data.provid} Tier1: GLS best={best_gls:.3f}hr "
        f"(cont={gls_cont:.2f}) power={gls_max:.3f}, "
        f"MBLS best={best_mbls:.3f}hr (cont={mbls_cont:.2f}) → PASS"
    )

    return Tier1Result(
        provid=data.provid, passes=True,
        best_period_gls=best_gls, best_period_mbls=best_mbls,
        gls_power_max=gls_max, snr=data.snr, n_obs=data.n_obs,
        reject_reason=None,
        test_periods=test_periods, gls_power=gls_pow, mbls_power=mbls_pow,
        window_power=window_pow,
        gls_contamination=gls_cont,
        mbls_contamination=mbls_cont,
    )


# ── GLS implementation ────────────────────────────────────────────────────────

def gls_power(
    t:      np.ndarray,
    y:      np.ndarray,
    dy:     np.ndarray,
    period: float,
) -> float:
    """
    Generalised Lomb-Scargle power at a single trial period.
    Zechmeister & Kürster (2009). Fits y = a*cos + b*sin + c via
    weighted least squares with a floating mean.
    """
    w  = 1.0 / dy**2
    W  = w.sum()
    ph = 2.0 * np.pi * t / period
    C, S = np.cos(ph), np.sin(ph)

    YY = np.sum(w * y**2) / W - (np.sum(w * y) / W)**2
    YC = np.sum(w * y * C) / W - (np.sum(w * y) / W) * (np.sum(w * C) / W)
    YS = np.sum(w * y * S) / W - (np.sum(w * y) / W) * (np.sum(w * S) / W)
    CC = np.sum(w * C**2)  / W - (np.sum(w * C) / W)**2
    SS = np.sum(w * S**2)  / W - (np.sum(w * S) / W)**2
    CS = np.sum(w * C * S) / W - (np.sum(w * C) / W) * (np.sum(w * S) / W)

    D = CC * SS - CS**2
    if D == 0 or YY == 0:
        return 0.0

    power = (SS * YC**2 - 2.0 * CS * YC * YS + CC * YS**2) / (YY * D)
    return float(np.clip(power, 0.0, 1.0))


def gls_periodogram(
    t:            np.ndarray,
    y:            np.ndarray,
    dy:           np.ndarray,
    test_periods: np.ndarray,
) -> np.ndarray:
    """GLS power over a grid of trial periods."""
    return np.array([gls_power(t, y, dy, p) for p in test_periods])


# ── MBLS implementation ───────────────────────────────────────────────────────

def mbls_periodogram(
    t:            np.ndarray,
    y:            np.ndarray,
    dy:           np.ndarray,
    bands:        np.ndarray,
    test_periods: np.ndarray,
    nterms:       int = 1,
) -> np.ndarray:
    """
    Multi-band Lomb-Scargle periodogram using gatspy.
    VanderPlas & Ivezić (2015).

    Expects raw (non-offset-corrected) magnitudes. MBLS fits a shared
    period with separate mean and amplitude per band — the per-band mean
    is part of the model, so pre-subtracting it removes information.

    Parameters
    ----------
    y      : raw magnitudes (geometry-corrected if available, no band offsets)
    bands  : band label per observation
    nterms : Fourier terms. 1 = single sinusoid (Tier 1), 2 = double hump (Tier 2)
    """
    model = LombScargleMultiband(Nterms_base=nterms, Nterms_band=0)
    model.fit(t, y, dy, bands)
    return model.periodogram(test_periods)


# ── Helper ────────────────────────────────────────────────────────────────────

def _reject(
    data:   PreparedData,
    config: PipelineConfig,
    reason: str,
) -> Tier1Result:
    empty = np.array([])
    logger.debug(f"{data.provid} Tier1: REJECT — {reason}")
    return Tier1Result(
        provid=data.provid, passes=False,
        best_period_gls=np.nan, best_period_mbls=np.nan,
        gls_power_max=0.0, snr=data.snr, n_obs=data.n_obs,
        reject_reason=reason,
        test_periods=empty, gls_power=empty, mbls_power=empty,
        window_power=empty,
        gls_contamination=np.nan,
        mbls_contamination=np.nan,
    )
