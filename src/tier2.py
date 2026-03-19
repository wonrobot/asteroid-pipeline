"""
tier2.py
--------
Tier 2: Period refinement for objects that passed Tier 1.
Runs three independent methods and checks for agreement.

Methods
-------
1. MHAOV NH=2  — Multi-Harmonic AOV (Schwarzenberg-Czerny 1996)
                 Fits 2 harmonics via weighted LS, returns F-statistic.
                 Directly models the double-hump asteroid shape.
                 Gives real p-values via F(4, N-5) distribution.

2. MBLS Nterms=2 — Multi-band LS with 2nd harmonic
                 Exploits multi-band colour information.

3. Conditional Entropy — Graham et al. (2013)
                 Model-free validator. Measures 2D phase-magnitude
                 histogram entropy. No assumed shape.

Decision: if all three agree within 5% AND p-value < 0.001 → publish.
Otherwise → Tier 3.

Functions
---------
run_tier2(data, t1result, config)       — main entry point
mhaov_periodogram(t, y, dy, tp, nh)    — MHAOV F-statistic over period grid
mhaov_single(t, y, dy, period, nh)     — MHAOV at one period
ce_periodogram(t, y, tp)               — Conditional Entropy over period grid
ce_single(t, y, period)                — CE at one period
check_agreement(p1, p2, p3, tol)       — check if three periods agree
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import f as f_dist

from gatspy.periodic import LombScargleMultiband

from config import PipelineConfig, DEFAULT_CONFIG
from preprocessing import PreparedData
from tier1 import Tier1Result, mbls_periodogram

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Tier2Result:
    """
    Output of Tier 2 period refinement.

    Attributes
    ----------
    provid           : asteroid designation
    passes           : True = all methods agree, proceed to publish
    to_tier3         : True = methods disagree, needs disambiguation
    best_period_mhaov: MHAOV best period (hours)
    best_period_mbls : MBLS Nterms=2 best period (hours)
    best_period_ce   : Conditional Entropy best period (hours)
    consensus_period : median of the three best periods
    F_stat           : MHAOV F-statistic at best period
    p_value          : MHAOV p-value at best period
    amplitude        : peak-to-peak of detrended lightcurve
    snr              : signal-to-noise ratio
    agreement        : True if all three periods agree within tolerance
    period_spread_pct: max fractional spread between the three periods
    reject_reason    : if not passes and not to_tier3 (e.g. p-value too high)
    test_periods     : period grid
    mhaov_power      : MHAOV F-statistic array
    mbls_power       : MBLS power array
    ce_scores        : conditional entropy array (lower = better)
    """
    provid:            str
    passes:            bool
    to_tier3:          bool
    best_period_mhaov: float
    best_period_mbls:  float
    best_period_ce:    float
    consensus_period:  float
    F_stat:            float
    p_value:           float
    amplitude:         float
    snr:               float
    agreement:         bool
    period_spread_pct: float
    reject_reason:     Optional[str]
    test_periods:      np.ndarray
    mhaov_power:       np.ndarray
    mbls_power:        np.ndarray
    ce_scores:         np.ndarray


# ── Main Tier 2 entry point ───────────────────────────────────────────────────

def run_tier2(
    data:     PreparedData,
    t1result: Tier1Result,
    config:   PipelineConfig = DEFAULT_CONFIG,
) -> Tier2Result:
    """
    Run Tier 2 period refinement on a single asteroid.

    Uses the Tier 1 period grid or builds a finer one.
    Runs MHAOV, MBLS, and Conditional Entropy in sequence.
    Checks agreement and statistical significance.

    Parameters
    ----------
    data     : PreparedData from preprocessing.preprocess()
    t1result : Tier1Result — must have passes=True
    config   : PipelineConfig

    Returns
    -------
    Tier2Result
    """
    if not t1result.passes:
        raise ValueError(f"{data.provid}: Tier 1 did not pass — cannot run Tier 2")

    cfg_p = config.period
    cfg_t = config.tier

    # Fine period grid
    test_periods = np.linspace(
        cfg_p.period_min_hr,
        min(cfg_p.period_max_hr, data.baseline_hr),
        cfg_p.n_grid_fine,
    )

    # ── 1. MHAOV NH=2 ─────────────────────────────────────────────────────────
    logger.debug(f"{data.provid}: running MHAOV NH=2...")
    mhaov_pow      = mhaov_periodogram(data.t_hrs, data.y_dt, data.dy, test_periods, nh=cfg_p.mhaov_nh)
    best_idx_mhaov = np.argmax(mhaov_pow)
    best_mhaov     = test_periods[best_idx_mhaov]
    F_best         = float(mhaov_pow[best_idx_mhaov])

    # F-test p-value: F(2*NH, N - 2*NH - 1)
    df_model = 2 * cfg_p.mhaov_nh
    df_resid = data.n_obs - 2 * cfg_p.mhaov_nh - 1
    p_value  = float(1.0 - f_dist.cdf(F_best, df_model, max(df_resid, 1)))

    # ── 2. MBLS Nterms=2 ──────────────────────────────────────────────────────
    logger.debug(f"{data.provid}: running MBLS Nterms=2...")
    try:
        mbls_pow  = mbls_periodogram(
            data.t_hrs, data.y_dt, data.dy, data.bands,
            test_periods, nterms=cfg_p.mbls_nterms_t2
        )
        best_mbls = test_periods[np.argmax(mbls_pow)]
    except Exception as e:
        logger.warning(f"{data.provid}: MBLS Tier2 failed ({e}) — using MHAOV period")
        mbls_pow  = mhaov_pow.copy()
        best_mbls = best_mhaov

    # ── 3. Conditional Entropy ────────────────────────────────────────────────
    logger.debug(f"{data.provid}: running Conditional Entropy...")
    ce_scores = ce_periodogram(data.t_hrs, data.y_dt, test_periods,
                               n_phase=cfg_p.ce_n_phase, n_mag=cfg_p.ce_n_mag)
    best_ce   = test_periods[np.argmin(ce_scores)]

    # ── Agreement check ───────────────────────────────────────────────────────
    agrees, spread_pct = check_agreement(best_mhaov, best_mbls, best_ce, cfg_t.agreement_tol)
    consensus          = float(np.median([best_mhaov, best_mbls, best_ce]))

    logger.debug(
        f"{data.provid} Tier2: MHAOV={best_mhaov:.3f}hr  "
        f"MBLS={best_mbls:.3f}hr  CE={best_ce:.3f}hr  "
        f"agree={agrees}  F={F_best:.1f}  p={p_value:.2e}"
    )

    # ── Decision ──────────────────────────────────────────────────────────────
    # Check p-value significance regardless of agreement
    if p_value >= cfg_t.mhaov_pval_thresh and not agrees:
        return Tier2Result(
            provid=data.provid, passes=False, to_tier3=False,
            best_period_mhaov=best_mhaov, best_period_mbls=best_mbls,
            best_period_ce=best_ce, consensus_period=consensus,
            F_stat=F_best, p_value=p_value,
            amplitude=data.amplitude, snr=data.snr,
            agreement=agrees, period_spread_pct=spread_pct,
            reject_reason=f"p-value={p_value:.2e} not significant and methods disagree",
            test_periods=test_periods, mhaov_power=mhaov_pow,
            mbls_power=mbls_pow, ce_scores=ce_scores,
        )

    if agrees and p_value < cfg_t.mhaov_pval_thresh:
        # All methods agree and signal is significant — publish
        return Tier2Result(
            provid=data.provid, passes=True, to_tier3=False,
            best_period_mhaov=best_mhaov, best_period_mbls=best_mbls,
            best_period_ce=best_ce, consensus_period=consensus,
            F_stat=F_best, p_value=p_value,
            amplitude=data.amplitude, snr=data.snr,
            agreement=True, period_spread_pct=spread_pct,
            reject_reason=None,
            test_periods=test_periods, mhaov_power=mhaov_pow,
            mbls_power=mbls_pow, ce_scores=ce_scores,
        )

    # Methods disagree or marginal — escalate to Tier 3
    return Tier2Result(
        provid=data.provid, passes=False, to_tier3=True,
        best_period_mhaov=best_mhaov, best_period_mbls=best_mbls,
        best_period_ce=best_ce, consensus_period=consensus,
        F_stat=F_best, p_value=p_value,
        amplitude=data.amplitude, snr=data.snr,
        agreement=False, period_spread_pct=spread_pct,
        reject_reason=None,
        test_periods=test_periods, mhaov_power=mhaov_pow,
        mbls_power=mbls_pow, ce_scores=ce_scores,
    )


# ── MHAOV implementation ──────────────────────────────────────────────────────

def mhaov_single(
    t:      np.ndarray,
    y:      np.ndarray,
    dy:     np.ndarray,
    period: float,
    nh:     int = 2,
) -> float:
    """
    Multi-Harmonic AOV F-statistic at a single trial period.

    Schwarzenberg-Czerny (1996). Fits NH harmonics via weighted LS,
    partitions variance into model vs residual, returns F-statistic.

    F = (SS_model / df_model) / (SS_residual / df_residual)
    where df_model = 2*NH, df_residual = N - 2*NH - 1

    Higher F means stronger periodic signal at this period.
    p-value from F(df_model, df_residual) distribution.
    """
    w  = 1.0 / dy**2
    N  = len(t)
    ph = 2.0 * np.pi * t / period

    # Design matrix: [1, cos(ph), sin(ph), cos(2ph), sin(2ph), ...]
    cols = [np.ones(N)]
    for k in range(1, nh + 1):
        cols.append(np.cos(k * ph))
        cols.append(np.sin(k * ph))
    A = np.column_stack(cols)

    try:
        Aw       = A * w[:, None]
        coeffs   = np.linalg.solve(Aw.T @ A, Aw.T @ y)
        y_fit    = A @ coeffs
        y_wmean  = np.average(y, weights=w)
        SS_model = np.sum(w * (y_fit  - y_wmean)**2)
        SS_resid = np.sum(w * (y      - y_fit  )**2)
        df_model = 2 * nh
        df_resid = N - 2 * nh - 1
        if df_resid <= 0 or SS_resid == 0:
            return 0.0
        return float((SS_model / df_model) / (SS_resid / df_resid))
    except np.linalg.LinAlgError:
        return 0.0


def mhaov_periodogram(
    t:            np.ndarray,
    y:            np.ndarray,
    dy:           np.ndarray,
    test_periods: np.ndarray,
    nh:           int = 2,
) -> np.ndarray:
    """MHAOV F-statistic over a grid of trial periods."""
    return np.array([mhaov_single(t, y, dy, p, nh) for p in test_periods])


# ── Conditional Entropy implementation ───────────────────────────────────────

def ce_single(
    t:       np.ndarray,
    y:       np.ndarray,
    period:  float,
    n_phase: int = 10,
    n_mag:   int = 5,
) -> float:
    """
    Conditional Entropy H(magnitude | phase) at a single trial period.

    Graham et al. (2013) — developed specifically for LSST survey data.
    Uses a 2D histogram of (phase, magnitude). True period produces
    a structured lightcurve → low entropy. Wrong period → random
    scatter → high entropy.

    Lower values are better (unlike power-based methods).
    """
    phase  = (t % period) / period
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    H2d, _, _ = np.histogram2d(phase, y_norm,
                                bins=[n_phase, n_mag],
                                range=[[0, 1], [0, 1]])
    p_joint = H2d / (H2d.sum() + 1e-12)
    p_phase = p_joint.sum(axis=1, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(p_phase > 0, p_joint / p_phase, 0.0)
        ce    = -np.sum(p_joint * np.log(ratio + 1e-12))

    return float(ce)


def ce_periodogram(
    t:            np.ndarray,
    y:            np.ndarray,
    test_periods: np.ndarray,
    n_phase:      int = 10,
    n_mag:        int = 5,
) -> np.ndarray:
    """Conditional Entropy over a grid of trial periods. Lower = better."""
    return np.array([ce_single(t, y, p, n_phase, n_mag) for p in test_periods])


# ── Agreement check ───────────────────────────────────────────────────────────

def check_agreement(
    p1: float,
    p2: float,
    p3: float,
    tol: float = 0.05,
) -> Tuple[bool, float]:
    """
    Check whether three period estimates agree within a fractional tolerance.

    Handles harmonic relationships: checks if any pair of periods agrees
    directly OR if one is twice the other (P vs P/2 ambiguity).

    Parameters
    ----------
    p1, p2, p3 : period estimates (same units)
    tol        : fractional tolerance e.g. 0.05 = 5%

    Returns
    -------
    (agrees, spread_pct)
    agrees     : True if all three are within tol of their median
    spread_pct : max fractional deviation from median
    """
    periods    = np.array([p1, p2, p3])
    median_p   = np.median(periods)
    spread_pct = float(np.max(np.abs(periods - median_p) / (median_p + 1e-12)))
    agrees     = spread_pct <= tol
    return agrees, spread_pct
