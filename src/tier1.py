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

# ── GLS false alarm probability (Zechmeister & Kürster 2009) ─────────────────
# Spin barrier for gravitationally bound (rubble-pile) asteroids.
# Asteroids larger than ~150m rotating faster than this would disrupt.
# Citation: Pravec & Harris (2000), Icarus 148, 12–20; Holsapple (2007).
SPIN_BARRIER_HR = 2.2

def gls_fap(
    power_max:   float,
    n_obs:       int,
    baseline_hr: float,
    p_min_hr:    float = 0.5,
    p_max_hr:    float = 24.0,
) -> float:
    """
    Analytical false alarm probability for the maximum GLS power on a grid.

    Uses the Zechmeister & Kürster (2009) single-frequency tail probability:
        P(z > z₀ | H₀) = (1 - z₀)^((N-3)/2)    [ZK09 Eq. 13]

    Combined over M independent frequencies (Horne & Baliunas 1986):
        FAP = 1 - (1 - p_single)^M
        M ≈ T_baseline × (f_max - f_min)

    Parameters
    ----------
    power_max   : maximum GLS power ∈ [0, 1]
    n_obs       : number of observations
    baseline_hr : observing baseline in hours
    p_min_hr    : lower period bound of search grid (hours)
    p_max_hr    : upper period bound of search grid (hours)

    Returns
    -------
    FAP ∈ [0, 1]. Low FAP → signal is significant → coarse grid is reliable.
    High FAP → signal is not significant → expand to fine grid warranted.
    """
    df_resid = max(n_obs - 3, 1)
    p_single = float((1.0 - power_max) ** (df_resid / 2.0))

    # Number of independent frequencies (Horne & Baliunas 1986)
    f_max = 1.0 / max(p_min_hr, 1e-6)
    f_min = 1.0 / max(p_max_hr, 1e-6)
    M     = max(1.0, baseline_hr * (f_max - f_min))

    fap = 1.0 - (1.0 - p_single) ** M
    return float(np.clip(fap, 0.0, 1.0))



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
    mbls_peaks          : top-N window-penalised MBLS peak periods (hours),
                          sorted by adjusted power descending.  Tier 2 uses
                          min(mbls_peaks) to set its grid lower bound so that
                          all significant T1 peaks — not just the argmax — are
                          covered at fine resolution.  Always length ≥ 1.
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
    mbls_peaks:          np.ndarray   # top-N window-penalised MBLS peaks (hrs)
    t1_pass2_trigger:    Optional[str]  # "A-spin_barrier"|"B-insignificant_fap"|"C-low_band_support"|None


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

    # ── Period grid — two-pass strategy ─────────────────────────────────────
    #
    # Pass 1: coarse grid from 0.5hr (2000 pts, fast ~0.5s for all objects).
    #         Screens for normal rotators (P > 0.5hr) at low cost.
    #         Window function computed on coarse grid to qualify Pass 2.
    #
    # Pass 2: expand to the Eyer & Bartholdi period floor (data.period_min_hr,
    #         typically ~0.023 hr for Rubin) when the coarse result suggests
    #         the true period may be below 0.5hr.
    #
    #         Trigger conditions (any one):
    #           (a) near_boundary: best coarse period < 0.75hr — could be P/2
    #           (b) weak_power: max coarse power < 0.15 — signal may be hiding
    #               below 0.5hr (but only if coarse best is NOT window-
    #               contaminated — low power from an alias should not trigger
    #               a fast-rotator search)
    #
    #         For Rubin data, data.period_min_hr ≈ 0.023 hr (Eyer & Bartholdi),
    #         so can_search_fast is almost always True and the decision reduces
    #         to whether the coarse result warrants expansion.
    #
    #         Grid density follows Greenstreet et al. (2026) Eq. 4:
    #           n = 5 × T × (1/P_min − 1/P_max)   [5× for T1 speed]
    #         Tier 2 uses 100× for full precision.
    p_max_t1    = min(cfg_p.period_max_hr, data.baseline_hr)
    FAST_THRESH = 0.5  # hr — coarse grid lower bound (sub-0.5hr needs fine grid)

    # ── Pass 1: coarse grid from 0.5hr ───────────────────────────────────────
    p_min_coarse    = max(data.period_min_hr, FAST_THRESH)
    test_periods    = np.linspace(p_min_coarse, p_max_t1, cfg_p.n_grid_coarse)
    gls_pow_coarse  = gls_periodogram(data.t_hrs, data.y_dt, data.dy, test_periods)
    best_coarse     = test_periods[np.argmax(gls_pow_coarse)]
    max_pow_coarse  = float(gls_pow_coarse.max())

    # Window function on coarse grid — qualifies Pass 2 trigger
    window_pow_coarse   = compute_window_function(data.t_hrs, test_periods)
    coarse_cont         = contamination_score(
        best_coarse, test_periods, window_pow_coarse,
        baseline_hr=data.baseline_hr,
    )
    coarse_contaminated = coarse_cont > CONTAMINATION_THRESHOLD

    # ── Pass 2 triggers — both have scientific citations ─────────────────────
    #
    # Trigger A — near spin barrier (Pravec & Harris 2000):
    #   If the coarse best period is below the rubble-pile spin barrier
    #   (2.2hr), the object may be a monolithic fast rotator. We expand to
    #   the Eyer & Bartholdi floor to confirm or refute sub-barrier rotation.
    #   The old threshold (FAST_THRESH × 1.5 = 0.75hr) was purely geometric
    #   (1.5× the grid floor) with no physical motivation.
    #
    # Trigger B — coarse GLS not significant (Zechmeister & Kürster 2009):
    #   If the maximum coarse GLS power does not exceed the FAP threshold,
    #   there is no evidence that the true period lies in the coarse range.
    #   Expanding to the fine grid is warranted to search for stronger fast-
    #   rotator signals below 0.5hr. The old threshold (power < 0.15) was
    #   a fixed scalar independent of N and baseline, giving FAP values that
    #   ranged from ~0.001 (N=150) to ~0.9 (N=30) — not a stable criterion.
    #
    # Trigger B is suppressed when the coarse best sits on a window peak
    # (alias suppression scenario — low power is not evidence of a fast
    # rotator, it is evidence that the real signal was aliased away).
    # ── Pass 2 triggers — all have scientific citations ──────────────────────
    #
    # Trigger A — near spin barrier (Pravec & Harris 2000):
    #   If the coarse best period is below the rubble-pile spin barrier
    #   (2.2hr), the object may be a monolithic fast rotator. We expand to
    #   the Eyer & Bartholdi floor to confirm or refute sub-barrier rotation.
    #
    # Trigger B — coarse GLS not significant (Zechmeister & Kürster 2009):
    #   If the maximum coarse GLS power does not exceed the FAP threshold,
    #   there is no evidence that the true period lies in the coarse range.
    #   Expanding to the fine grid is warranted to search for stronger fast-
    #   rotator signals below 0.5hr. Suppressed when coarse best is already
    #   window-contaminated (alias suppression, not a fast rotator).
    #
    # Trigger C — low MBLS band support at coarse best (Change 10):
    #   A genuine slow rotator (P > 0.5hr) will produce consistent multi-band
    #   support at its period. An alias of an ultrafast signal (e.g. MJ71 at
    #   0.031hr aliasing to 18.8hr) may show high GLS power in the merged
    #   series but weak per-band coherence, because the alias structure is
    #   incoherent across band sampling patterns.
    #   We compute a quick MBLS band-support score at best_coarse and trigger
    #   Pass 2 if it is below the threshold — even when Trigger A and B are
    #   both false. This catches MJ71/MU15 class objects where a strong alias
    #   suppressed both FAP and spin-barrier triggers.
    #   Physical basis: VanderPlas & Ivezić (2015) show MBLS band support
    #   degrades for aliases relative to true periods.
    fap_coarse         = gls_fap(max_pow_coarse, data.n_obs, data.baseline_hr,
                                  p_min_hr=FAST_THRESH, p_max_hr=p_max_t1)
    near_spin_barrier  = best_coarse < SPIN_BARRIER_HR          # Pravec & Harris (2000)
    insignificant_coarse = fap_coarse > cfg_t.gls_fap_expand_thresh  # ZK09 FAP
    can_search_fast    = data.period_min_hr < FAST_THRESH        # floor supports sub-0.5hr

    # Trigger C: quick MBLS band-support at coarse best
    low_band_support_coarse = False
    if can_search_fast and not near_spin_barrier and not insignificant_coarse:
        # Only worth computing when Triggers A and B would NOT fire — this is
        # the novel case: strong FAP, not near spin barrier, but may be alias.
        try:
            from tier2 import compute_mbls_band_support
            _bs, _n_sup, _frac = compute_mbls_band_support(
                data.t_hrs, data.y_multiband, data.dy, data.bands,
                period=float(best_coarse), nterms=1,
            )
            low_band_support_coarse = _frac < cfg_t.t1_band_support_pass2_thresh
            if low_band_support_coarse:
                logger.debug(
                    f"{data.provid}: Pass 2 Trigger C — coarse best={best_coarse:.3f}hr "
                    f"has low MBLS band support ({_frac:.2f} < "
                    f"{cfg_t.t1_band_support_pass2_thresh}) — possible ultrafast alias"
                )
        except Exception as _e:
            logger.debug(f"{data.provid}: Trigger C band-support check failed ({_e}) — skipping")

    should_expand = (
        near_spin_barrier
        or (insignificant_coarse and not coarse_contaminated)
        or low_band_support_coarse
    ) and can_search_fast

    if coarse_contaminated and insignificant_coarse and not near_spin_barrier:
        logger.debug(
            f"{data.provid}: Pass 2 suppressed — coarse best={best_coarse:.3f}hr "
            f"is window-contaminated (cont={coarse_cont:.2f}); "
            f"low power is likely alias suppression, not a fast rotator"
        )

    pass2_trigger: Optional[str] = None
    if should_expand:
        pass2_trigger = ("A-spin_barrier" if near_spin_barrier
                         else "C-low_band_support" if low_band_support_coarse
                         else "B-insignificant_fap")
        trigger = pass2_trigger  # keep local alias for existing log lines
        # Greenstreet-style grid: 5× oversampling for T1 speed (Tier 2 uses 100×)
        n_fast = min(50_000, max(cfg_p.n_grid_coarse,
                     int(5 * data.baseline_hr
                         * (1.0/data.period_min_hr - 1.0/p_max_t1))))
        test_periods = np.linspace(data.period_min_hr, p_max_t1, n_fast)
        logger.debug(
            f"{data.provid}: Pass 2 expanded to {len(test_periods)} pts "
            f"(floor={data.period_min_hr*60:.1f}min, trigger={trigger}, "
            f"near_spin_barrier={near_spin_barrier}, insignificant_fap={fap_coarse:.3f}, "
            f"coarse_cont={coarse_cont:.2f})"
        )
    # else: keep coarse test_periods — normal rotator or alias-suppressed

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
    gls_cont     = contamination_score(best_gls, test_periods, window_pow,
                                   baseline_hr=data.baseline_hr)

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

    # ── Window-penalised MBLS peaks ───────────────────────────────────────────
    # Same treatment as GLS: find the top N peaks in the window-adjusted MBLS
    # power. This gives Tier 2 a ranked list of candidate periods rather than
    # a single argmax, so its search grid covers all significant peaks and their
    # harmonics — preventing T1 alias-lock from misplacing the T2 grid.
    #
    # N_MBLS_PEAKS=5: captures fundamental + up to 4 alias/harmonic peaks.
    # Tier 2 uses min(mbls_peaks)/4 as its grid lower bound.
    N_MBLS_PEAKS = 5
    mbls_peak_periods, _, _ = window_informed_peaks(
        test_periods, mbls_pow, window_pow,
        n_peaks=N_MBLS_PEAKS,
        min_period_hr=float(test_periods[0]),
        max_period_hr=float(test_periods[-1]),
    )
    # Guarantee at least the argmax is present (defensive fallback)
    if len(mbls_peak_periods) == 0:
        mbls_peak_periods = np.array([best_mbls])

    # Score MBLS best period for window contamination
    mbls_cont = contamination_score(float(best_mbls), test_periods, window_pow,
                                baseline_hr=data.baseline_hr)

    if mbls_cont > CONTAMINATION_THRESHOLD:
        logger.debug(
            f"{data.provid}: MBLS best={best_mbls:.3f}hr is window-contaminated "
            f"(score={mbls_cont:.2f})"
        )

    # ── Summary log ──────────────────────────────────────────────────────────
    logger.debug(
        f"{data.provid} Tier1: GLS best={best_gls:.3f}hr "
        f"(cont={gls_cont:.2f}) power={gls_max:.3f}, "
        f"MBLS best={best_mbls:.3f}hr (cont={mbls_cont:.2f}) "
        f"peaks={[f'{p:.3f}' for p in mbls_peak_periods]} → PASS"
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
        mbls_peaks=mbls_peak_periods,
        t1_pass2_trigger=pass2_trigger,
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
        mbls_peaks=empty,
        t1_pass2_trigger=None,
    )
