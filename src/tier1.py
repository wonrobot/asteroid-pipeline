"""
tier1.py
--------
Tier 1: Fast screening of all asteroids.
Runs GLS + MBLS Nterms=1 on every object that passes quality cuts.

Goal: eliminate objects with no detectable periodicity quickly.
Cost: ~milliseconds per asteroid (parallelisable).

Functions
---------
run_tier1(data, config)         — run Tier 1 on a PreparedData object
gls_power(t, y, dy, period)    — Generalised Lomb-Scargle power at one period
gls_periodogram(t, y, dy, tp)  — GLS over a period grid
mbls_periodogram(t,y,dy,b,tp)  — Multi-band LS over a period grid
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from gatspy.periodic import LombScargleMultiband

from config import PipelineConfig, DEFAULT_CONFIG
from preprocessing import PreparedData

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Tier1Result:
    """
    Output of Tier 1 screening.

    Attributes
    ----------
    provid          : asteroid designation
    passes          : True if this object should proceed to Tier 2
    best_period_gls : best period from GLS (hours) — note: often finds P/2
    best_period_mbls: best period from MBLS Nterms=1 (hours)
    gls_power_max   : peak GLS power (0–1)
    snr             : amplitude SNR from preprocessing
    n_obs           : number of observations
    reject_reason   : if passes=False, explains why (else None)
    test_periods    : period grid used
    gls_power       : full GLS power array (for plotting)
    mbls_power      : full MBLS power array (for plotting)
    """
    provid:           str
    passes:           bool
    best_period_gls:  float
    best_period_mbls: float
    gls_power_max:    float
    snr:              float
    n_obs:            int
    reject_reason:    Optional[str]
    test_periods:     np.ndarray
    gls_power:        np.ndarray
    mbls_power:       np.ndarray


# ── Main Tier 1 entry point ───────────────────────────────────────────────────

def run_tier1(
    data: PreparedData,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tier1Result:
    """
    Run Tier 1 screening on a single prepared asteroid.

    Checks:
    1. Minimum observation count
    2. Minimum SNR
    3. GLS peak power > 0.1 (weak periodicity filter)

    Parameters
    ----------
    data   : PreparedData from preprocessing.preprocess()
    config : PipelineConfig

    Returns
    -------
    Tier1Result — check .passes to decide if Tier 2 should run
    """
    cfg_t  = config.tier
    cfg_p  = config.period

    # ── Hard quality gates (no computation needed) ────────────────────────────
    if data.n_obs < cfg_t.min_obs:
        return _reject(data, config,
                       f"n_obs={data.n_obs} < min_obs={cfg_t.min_obs}")

    if data.snr < cfg_t.snr_threshold:
        return _reject(data, config,
                       f"SNR={data.snr:.2f} < threshold={cfg_t.snr_threshold}")

    # ── Period grid ───────────────────────────────────────────────────────────
    test_periods = np.linspace(
        cfg_p.period_min_hr,
        min(cfg_p.period_max_hr, data.baseline_hr),
        cfg_p.n_grid_coarse,
    )

    # ── GLS ───────────────────────────────────────────────────────────────────
    gls_pow  = gls_periodogram(data.t_hrs, data.y_dt, data.dy, test_periods)
    best_gls = test_periods[np.argmax(gls_pow)]
    gls_max  = float(gls_pow.max())

    # ── MBLS Nterms=1 ─────────────────────────────────────────────────────────
    try:
        mbls_pow  = mbls_periodogram(
            data.t_hrs, data.y_dt, data.dy, data.bands,
            test_periods, nterms=config.period.mbls_nterms_t1
        )
        best_mbls = test_periods[np.argmax(mbls_pow)]
    except Exception as e:
        logger.warning(f"{data.provid}: MBLS Tier1 failed ({e}) — using GLS only")
        mbls_pow  = gls_pow.copy()
        best_mbls = best_gls

    # ── Weak periodicity filter ───────────────────────────────────────────────
    # GLS power < 0.05 almost certainly means no periodic signal is detectable
    if gls_max < 0.05:
        return Tier1Result(
            provid=data.provid, passes=False,
            best_period_gls=best_gls, best_period_mbls=best_mbls,
            gls_power_max=gls_max, snr=data.snr, n_obs=data.n_obs,
            reject_reason=f"GLS power={gls_max:.3f} too low — no periodic signal",
            test_periods=test_periods, gls_power=gls_pow, mbls_power=mbls_pow,
        )

    logger.debug(
        f"{data.provid} Tier1: GLS best={best_gls:.3f}hr "
        f"power={gls_max:.3f}, MBLS best={best_mbls:.3f}hr  → PASS"
    )

    return Tier1Result(
        provid=data.provid, passes=True,
        best_period_gls=best_gls, best_period_mbls=best_mbls,
        gls_power_max=gls_max, snr=data.snr, n_obs=data.n_obs,
        reject_reason=None,
        test_periods=test_periods, gls_power=gls_pow, mbls_power=mbls_pow,
    )


# ── GLS implementation ────────────────────────────────────────────────────────

def gls_power(
    t:  np.ndarray,
    y:  np.ndarray,
    dy: np.ndarray,
    period: float,
) -> float:
    """
    Generalised Lomb-Scargle power at a single trial period.

    Zechmeister & Kürster (2009) — fits y = a*cos + b*sin + c
    via weighted least squares. Correctly handles a floating mean,
    unlike standard LS which forces zero mean.

    Parameters
    ----------
    t      : time array (hours)
    y      : magnitude offsets (detrended)
    dy     : photometric uncertainties
    period : trial period (hours)

    Returns
    -------
    float in [0, 1] — fractional variance explained by the sinusoid
    """
    w  = 1.0 / dy**2
    W  = w.sum()
    ph = 2.0 * np.pi * t / period
    C, S = np.cos(ph), np.sin(ph)

    # Weighted sums (notation from Zechmeister & Kürster 2009)
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
    t:           np.ndarray,
    y:           np.ndarray,
    dy:          np.ndarray,
    test_periods: np.ndarray,
) -> np.ndarray:
    """
    GLS power over a grid of trial periods.

    Returns
    -------
    np.ndarray of power values, same length as test_periods
    """
    return np.array([gls_power(t, y, dy, p) for p in test_periods])


# ── MBLS implementation ───────────────────────────────────────────────────────

def mbls_periodogram(
    t:           np.ndarray,
    y:           np.ndarray,
    dy:          np.ndarray,
    bands:       np.ndarray,
    test_periods: np.ndarray,
    nterms:      int = 1,
) -> np.ndarray:
    """
    Multi-band Lomb-Scargle periodogram using gatspy.

    VanderPlas & Ivezić (2015) — fits a shared period with per-band
    amplitude and phase offsets. With Nterms=1 this is the fast
    single-harmonic version used in Tier 1.

    Parameters
    ----------
    nterms : int
        Number of Fourier terms. 1 = single sinusoid, 2 = double hump.

    Returns
    -------
    np.ndarray of power values, same length as test_periods
    """
    model = LombScargleMultiband(Nterms_base=nterms, Nterms_band=1)
    model.fit(t, y, dy, bands)
    return model.periodogram(test_periods)


# ── Helper ────────────────────────────────────────────────────────────────────

def _reject(
    data: PreparedData,
    config: PipelineConfig,
    reason: str,
) -> Tier1Result:
    """Create a failed Tier1Result with empty arrays."""
    empty = np.array([])
    logger.debug(f"{data.provid} Tier1: REJECT — {reason}")
    return Tier1Result(
        provid=data.provid, passes=False,
        best_period_gls=np.nan, best_period_mbls=np.nan,
        gls_power_max=0.0, snr=data.snr, n_obs=data.n_obs,
        reject_reason=reason,
        test_periods=empty, gls_power=empty, mbls_power=empty,
    )
