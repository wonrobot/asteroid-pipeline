"""
tier2.py
--------
Tier 2: Period refinement on a fine frequency grid.

Architecture
------------
Two independent period-finding methods run on the same fine grid:

  MBLS (VanderPlas & Ivezic 2015)
    Multi-band Lomb-Scargle, Nterms=2 (double-hump model).
    Primary detector. Fits a shared period with per-band amplitude
    and mean — exploits all colour information jointly.

  MHAOV (Schwarzenberg-Czerny 1996)
    Multi-Harmonic Analysis of Variance, NH=2-4 adaptive.
    Corroboration method. Collapses multi-band data to a single
    detrended series and applies ANOVA significance test.

Period-doubling decision
------------------------
MBLS raw period P_raw may be the half-period of the true rotation
(common for symmetric double-hump lightcurves where both minima are
equally deep). To resolve this we compare MHAOV's independent
estimate against P_raw:

  MHAOV ≈ 2 × P_raw  →  adopt 2 × P_raw as consensus
  otherwise           →  keep P_raw

This is the minimal, explainable rule. Two independent methods
finding periods in a 2:1 ratio is direct evidence of doubling.

Decision paths
--------------
  mbls_confirmed  : MHAOV confirms MBLS consensus (within tol or harmonic)
                    + at least one significance gate passes → publish
  mbls_sig_only   : MBLS significant, MHAOV does not confirm → publish R≤2
  to_tier3=True   : both gates pass but MHAOV finds a genuinely different
                    period → ambiguous, send to Tier 3
  reject          : neither gate significant → no detectable period

Conditional Entropy (CE, Cincotta et al. 1999) runs as annotation only
when the period floor supports it. It never participates in the decision.
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
from window import (
    compute_window_function,
    contamination_score,
    window_informed_peaks,
    CONTAMINATION_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Threshold for MBLS per-band chi-sq improvement to count a band as
# supporting the consensus period.
BAND_SUPPORT_THRESH = 0.10


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Tier2Result:
    """
    Output of Tier 2 period refinement.

    Attributes
    ----------
    provid               : asteroid designation
    passes               : True = methods agree, proceed to publish
    to_tier3             : True = methods disagree, needs disambiguation
    best_period_mhaov    : MHAOV best period (hours)
    best_period_mbls_raw : MBLS best period before doubling check (hours)
    best_period_mbls     : MBLS consensus period after doubling check (hours)
    best_period_ce       : CE best period (hours, NaN if not run)
    consensus_period     : adopted period = best_period_mbls
    was_doubled          : True if MHAOV indicated 2×P_raw
    F_stat               : MHAOV F-statistic at best period
    p_value              : MHAOV p-value at best period
    amplitude            : peak-to-peak amplitude (mag)
    snr                  : signal-to-noise ratio
    agreement            : "mbls_confirmed" | "mbls_sig_only" | False
    period_spread_pct    : |MHAOV - MBLS| / MBLS after doubling
    reject_reason        : explanation if passes=False and to_tier3=False
    test_periods         : period grid used
    mhaov_power          : MHAOV F-statistic array
    mbls_power           : MBLS power array
    ce_scores            : CE score array (NaN array if not run)
    window_power         : spectral window function on test_periods
    mhaov_contamination  : cadence-alias score for MHAOV period [0-1]
    mbls_contamination   : cadence-alias score for MBLS period [0-1]
    ce_contamination     : cadence-alias score for CE period [0-1]
    consensus_contamination : cadence-alias score for consensus [0-1]
    mbls_fap             : MBLS false alarm probability via permutation [0-1]
    mbls_sig             : True if mbls_fap < cfg.tier.mbls_fap_thresh
    mhaov_sig            : True if p_value < cfg.tier.mhaov_pval_thresh
    both_sig             : True if both significance gates passed
    mbls_band_support    : per-band chi-sq improvement dict
    mbls_n_bands_supporting : bands with improvement > BAND_SUPPORT_THRESH
    mbls_band_support_frac  : fraction of bands supporting [0-1]
    mhaov_confirms       : True if MHAOV agrees with consensus within tol
    mbls_top_periods     : top-5 window-penalised MBLS peak periods (hrs)
    mbls_top_powers      : MBLS power at each top-5 peak
    """
    provid:                  str
    passes:                  bool
    to_tier3:                bool
    best_period_mhaov:       float
    best_period_mbls_raw:    float
    best_period_mbls:        float
    best_period_ce:          float
    consensus_period:        float
    was_doubled:             bool
    F_stat:                  float
    p_value:                 float
    amplitude:               float
    snr:                     float
    agreement:               object
    period_spread_pct:       float
    reject_reason:           Optional[str]
    test_periods:            np.ndarray
    mhaov_power:             np.ndarray
    mbls_power:              np.ndarray
    ce_scores:               np.ndarray
    window_power:            np.ndarray
    mhaov_contamination:     float
    mbls_contamination:      float
    ce_contamination:        float
    consensus_contamination: float
    mbls_fap:                float
    mbls_sig:                bool
    mhaov_sig:               bool
    both_sig:                bool
    mbls_band_support:       dict
    mbls_n_bands_supporting: int
    mbls_band_support_frac:  float
    mhaov_confirms:          bool
    mbls_top_periods:        np.ndarray
    mbls_top_powers:         np.ndarray


# ── Main Tier 2 entry point ───────────────────────────────────────────────────

def run_tier2(
    data:     PreparedData,
    t1result: Tier1Result,
    config:   PipelineConfig = DEFAULT_CONFIG,
) -> Tier2Result:
    """
    Run Tier 2 period refinement on a fine frequency grid.

    Parameters
    ----------
    data     : preprocessed asteroid data
    t1result : Tier 1 result (provides period grid bounds via mbls_peaks)
    config   : pipeline configuration

    Returns
    -------
    Tier2Result
    """
    cfg_p = config.period
    cfg_t = config.tier
    empty = np.array([])

    # ── Fine period grid ──────────────────────────────────────────────────────
    # Lower bound: min of all significant T1 MBLS peaks / 4, so the fine
    # grid covers any sub-harmonic the coarse grid may have found.
    p_min = float(np.min(t1result.mbls_peaks)) / 4.0 if len(t1result.mbls_peaks) else data.period_min_hr
    p_min = max(p_min, data.period_min_hr)
    p_max = min(cfg_p.period_max_hr, data.baseline_hr)

    n_t2  = int(100 * data.baseline_hr * (1.0 / p_min - 1.0 / p_max))
    n_t2  = max(min(n_t2, 100_000), cfg_p.n_grid_coarse)
    test_periods = np.linspace(p_min, p_max, n_t2)

    # ── Window function ───────────────────────────────────────────────────────
    window_pow = compute_window_function(data.t_hrs, test_periods)

    # ── 1. MHAOV ──────────────────────────────────────────────────────────────
    logger.debug(f"{data.provid}: running MHAOV adaptive NH=2-4...")
    mhaov_pow, best_mhaov, F_best, best_nh = mhaov_periodogram_adaptive(
        data.t_hrs, data.y_dt, data.dy, test_periods,
        nh_min=cfg_p.mhaov_nh,
    )

    df_model = 2 * cfg_p.mhaov_nh
    df_resid = data.n_obs - 2 * cfg_p.mhaov_nh - 1
    p_value  = float(1.0 - f_dist.cdf(F_best, df_model, max(df_resid, 1)))

    mhaov_cont = contamination_score(best_mhaov, test_periods, window_pow,
                                     baseline_hr=data.baseline_hr)
    if mhaov_cont > CONTAMINATION_THRESHOLD:
        logger.warning(f"{data.provid}: MHAOV best={best_mhaov:.3f}hr window-contaminated "
                       f"(score={mhaov_cont:.2f})")

    # ── 2. MBLS ───────────────────────────────────────────────────────────────
    logger.debug(f"{data.provid}: running MBLS Nterms=2...")
    try:
        mbls_pow      = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods, nterms=cfg_p.mbls_nterms_t2,
        )
        best_mbls_raw = test_periods[np.argmax(mbls_pow)]
    except Exception as e:
        logger.warning(f"{data.provid}: MBLS failed ({e}) — falling back to MHAOV")
        mbls_pow      = mhaov_pow.copy()
        best_mbls_raw = best_mhaov

    mbls_cont = contamination_score(best_mbls_raw, test_periods, window_pow,
                                    baseline_hr=data.baseline_hr)
    if mbls_cont > CONTAMINATION_THRESHOLD:
        logger.warning(f"{data.provid}: MBLS best={best_mbls_raw:.3f}hr window-contaminated "
                       f"(score={mbls_cont:.2f})")

    # ── 3. Period-doubling decision ───────────────────────────────────────────
    # If MHAOV independently finds ~2×P_raw, the true period is the doubled
    # value. This is direct observational evidence from a second independent
    # method — no sub-fundamental test required.
    tol         = cfg_t.agreement_tol
    was_doubled = (
        abs(best_mhaov - 2.0 * best_mbls_raw) / (2.0 * best_mbls_raw + 1e-12) <= tol
    )
    best_mbls = 2.0 * best_mbls_raw if was_doubled else best_mbls_raw
    if was_doubled:
        logger.debug(f"{data.provid}: MHAOV={best_mhaov:.3f}hr ≈ 2×MBLS raw={best_mbls_raw:.3f}hr "
                     f"→ doubling to {best_mbls:.3f}hr")

    # ── 4. Top-5 MBLS peaks ───────────────────────────────────────────────────
    try:
        _top_ps, _top_pows = window_informed_peaks(
            test_periods, mbls_pow, window_pow,
            n_peaks=5, baseline_hr=data.baseline_hr,
        )
        _mbls_top_periods = np.array(_top_ps)
        _mbls_top_powers  = np.array(_top_pows)
    except Exception:
        _mbls_top_periods = np.array([best_mbls_raw])
        _mbls_top_powers  = np.array([float(np.max(mbls_pow))])

    # ── 5. CE (annotation only) ───────────────────────────────────────────────
    best_ce   = np.nan
    ce_scores = empty
    ce_cont   = np.nan
    if data.period_min_hr >= 1.0:
        try:
            ce_scores = ce_periodogram(data.t_hrs, data.y_dt, test_periods)
            best_ce   = test_periods[np.argmin(ce_scores)]
            ce_cont   = contamination_score(best_ce, test_periods, window_pow,
                                            baseline_hr=data.baseline_hr)
            logger.debug(f"{data.provid}: CE best={best_ce:.3f}hr (annotation only)")
        except Exception as e:
            logger.debug(f"{data.provid}: CE skipped ({e})")

    # ── 6. MBLS significance ──────────────────────────────────────────────────
    mbls_fap = compute_mbls_fap(
        data.t_hrs, data.y_multiband, data.dy, data.bands,
        test_periods, observed_max_power=float(np.max(mbls_pow)),
        nterms=cfg_p.mbls_nterms_t2,
        n_perm=cfg_t.mbls_fap_n_perm,
    )
    mbls_sig   = mbls_fap < cfg_t.mbls_fap_thresh
    mhaov_sig  = p_value  < cfg_t.mhaov_pval_thresh
    either_sig = mbls_sig or mhaov_sig
    both_sig   = mbls_sig and mhaov_sig

    # ── 7. Band support ───────────────────────────────────────────────────────
    try:
        band_support, n_bands_supporting, band_support_frac = compute_mbls_band_support(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            period=best_mbls, nterms=cfg_p.mbls_nterms_t2,
        )
    except Exception:
        band_support, n_bands_supporting, band_support_frac = {}, 0, 0.0

    # ── 8. Consensus and MHAOV confirmation ───────────────────────────────────
    consensus      = best_mbls
    consensus_cont = contamination_score(consensus, test_periods, window_pow,
                                         baseline_hr=data.baseline_hr)
    spread_pct     = abs(best_mhaov - consensus) / (consensus + 1e-12)

    def _confirms(mhaov_p, ref_p):
        return any(
            abs(mhaov_p - ref_p * m) / (ref_p * m + 1e-12) <= tol
            for m in [1.0, 0.5, 2.0]
        )
    mhaov_confirms = _confirms(best_mhaov, consensus)

    logger.debug(
        f"{data.provid}: consensus={consensus:.3f}hr  "
        f"MHAOV={'confirms' if mhaov_confirms else 'disagrees'}  "
        f"doubled={was_doubled}  mbls_fap={mbls_fap:.4f}  p={p_value:.2e}"
    )

    # ── 9. Decision ───────────────────────────────────────────────────────────
    def _result(passes, to_tier3, agreement_val, reject_reason=None):
        return Tier2Result(
            provid=data.provid,
            passes=passes,
            to_tier3=to_tier3,
            best_period_mhaov=best_mhaov,
            best_period_mbls_raw=best_mbls_raw,
            best_period_mbls=best_mbls,
            best_period_ce=best_ce,
            consensus_period=consensus,
            was_doubled=was_doubled,
            F_stat=F_best,
            p_value=p_value,
            amplitude=data.amplitude,
            snr=data.snr,
            agreement=agreement_val,
            period_spread_pct=spread_pct,
            reject_reason=reject_reason,
            test_periods=test_periods,
            mhaov_power=mhaov_pow,
            mbls_power=mbls_pow,
            ce_scores=ce_scores,
            window_power=window_pow,
            mhaov_contamination=mhaov_cont,
            mbls_contamination=mbls_cont,
            ce_contamination=ce_cont,
            consensus_contamination=consensus_cont,
            mbls_fap=mbls_fap,
            mbls_sig=mbls_sig,
            mhaov_sig=mhaov_sig,
            both_sig=both_sig,
            mbls_band_support=band_support,
            mbls_n_bands_supporting=n_bands_supporting,
            mbls_band_support_frac=band_support_frac,
            mhaov_confirms=mhaov_confirms,
            mbls_top_periods=_mbls_top_periods,
            mbls_top_powers=_mbls_top_powers,
        )

    # Path 1: no signal
    if not mbls_sig and not mhaov_sig:
        return _result(
            passes=False, to_tier3=False, agreement_val=False,
            reject_reason=(
                f"neither gate significant "
                f"(MBLS FAP={mbls_fap:.4f}, MHAOV p={p_value:.2e})"
            ),
        )

    # Path 2: MHAOV confirms consensus → publish
    if mhaov_confirms and either_sig:
        logger.debug(f"{data.provid}: MHAOV confirms → mbls_confirmed")
        return _result(passes=True, to_tier3=False, agreement_val="mbls_confirmed")

    # Path 3: both significant but genuinely different periods → Tier 3
    if both_sig and not mhaov_confirms:
        logger.debug(f"{data.provid}: both significant, MHAOV disagrees → Tier 3")
        return _result(passes=False, to_tier3=True, agreement_val=False)

    # Path 4: MBLS significant alone → publish with lower confidence
    if mbls_sig:
        logger.debug(f"{data.provid}: MBLS significant, MHAOV disagrees → mbls_sig_only")
        return _result(passes=True, to_tier3=False, agreement_val="mbls_sig_only")

    # Path 5: MHAOV significant alone → Tier 3 for disambiguation
    return _result(passes=False, to_tier3=True, agreement_val=False)


# ── MBLS permutation FAP ──────────────────────────────────────────────────────

def compute_mbls_fap(
    t:                  np.ndarray,
    y:                  np.ndarray,
    dy:                 np.ndarray,
    bands:              np.ndarray,
    test_periods:       np.ndarray,
    observed_max_power: float,
    nterms:             int   = 2,
    n_perm:             int   = 200,
    n_coarse:           int   = 500,
    seed:               int   = 42,
) -> float:
    """
    MBLS false alarm probability via label permutation.

    Shuffles observation timestamps to build the null distribution of
    maximum power. FAP = fraction of permutations where max power
    exceeds the observed value.

    Returns
    -------
    fap : float in [0, 1]
    """
    rng    = np.random.default_rng(seed)
    p_lo   = float(test_periods[0])
    p_hi   = float(test_periods[-1])
    coarse = np.linspace(p_lo, p_hi, n_coarse)

    n_exceed = 0
    for _ in range(n_perm):
        t_perm = rng.permutation(t)
        try:
            pow_perm = mbls_periodogram(t_perm, y, dy, bands, coarse, nterms=nterms)
            if float(pow_perm.max()) >= observed_max_power:
                n_exceed += 1
        except Exception:
            pass

    return float(n_exceed) / float(n_perm)


# ── MBLS per-band support ─────────────────────────────────────────────────────

def compute_mbls_band_support(
    t:       np.ndarray,
    y:       np.ndarray,
    dy:      np.ndarray,
    bands:   np.ndarray,
    period:  float,
    nterms:  int = 2,
) -> tuple:
    """
    Per-band chi-sq improvement of MBLS fit at a given period.

    Measures how much better a Fourier model at `period` fits each
    band versus a flat (constant) model.

    Returns
    -------
    (band_support_dict, n_bands_supporting, support_frac)
    """
    model = LombScargleMultiband(Nterms_base=nterms, Nterms_band=0)
    model.fit(t, y, dy, bands)

    unique_bands  = np.unique(bands)
    band_support  = {}

    for b in unique_bands:
        mask = bands == b
        t_b, y_b, dy_b = t[mask], y[mask], dy[mask]
        if len(t_b) < 2 * nterms + 2:
            band_support[b] = 0.0
            continue
        w      = 1.0 / dy_b**2
        y_mean = np.average(y_b, weights=w)
        ss_tot = float(np.sum(w * (y_b - y_mean)**2))
        if ss_tot <= 0:
            band_support[b] = 0.0
            continue
        y_pred = model.predict(t_b, filts=np.full(len(t_b), b), period=period)
        ss_res = float(np.sum(w * (y_b - y_pred)**2))
        band_support[b] = max(0.0, 1.0 - ss_res / ss_tot)

    n_supporting = sum(1 for v in band_support.values() if v > BAND_SUPPORT_THRESH)
    support_frac = n_supporting / len(unique_bands) if unique_bands.size > 0 else 0.0

    return band_support, n_supporting, support_frac


# ── MHAOV ─────────────────────────────────────────────────────────────────────

def mhaov_single(
    t:      np.ndarray,
    y:      np.ndarray,
    dy:     np.ndarray,
    period: float,
    nh:     int = 2,
) -> float:
    """MHAOV F-statistic at a single trial period."""
    w      = 1.0 / dy**2
    phase  = (t / period) % 1.0
    n_bins = 2 * nh
    bins   = np.floor(phase * n_bins).astype(int) % n_bins

    w_tot   = w.sum()
    y_wmean = np.dot(w, y) / w_tot
    ss_tot  = float(np.sum(w * (y - y_wmean)**2))

    ss_model = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() < 1:
            continue
        w_b  = w[mask].sum()
        ym_b = np.dot(w[mask], y[mask]) / w_b
        ss_model += w_b * (ym_b - y_wmean)**2

    df_model = n_bins - 1
    df_resid = max(len(t) - n_bins, 1)
    ss_resid = max(ss_tot - ss_model, 0.0)

    if ss_resid <= 0:
        return np.inf
    return (ss_model / df_model) / (ss_resid / df_resid)


def mhaov_periodogram(
    t:            np.ndarray,
    y:            np.ndarray,
    dy:           np.ndarray,
    test_periods: np.ndarray,
    nh:           int = 2,
) -> np.ndarray:
    """MHAOV F-statistic array over all trial periods."""
    return np.array([mhaov_single(t, y, dy, p, nh) for p in test_periods])


def mhaov_periodogram_adaptive(
    t:             np.ndarray,
    y:             np.ndarray,
    dy:            np.ndarray,
    test_periods:  np.ndarray,
    nh_min:        int   = 2,
    nh_max:        int   = 4,    # retained for API compatibility, not used
    n_top_peaks:   int   = 10,   # retained for API compatibility, not used
    f_pval_thresh: float = 0.10, # retained for API compatibility, not used
) -> Tuple[np.ndarray, float, float, int]:
    """
    MHAOV periodogram at fixed NH=2.

    NH=2 uses 4 phase bins — sufficient to detect the standard
    double-hump asteroid lightcurve and directly comparable to
    MBLS Nterms=2. Both methods model the same harmonic complexity,
    so their period agreement is physically meaningful.

    Adaptive NH upgrade (NH=2→4) was removed because NH=4 (8 bins)
    peaks at different periods than NH=2 (4 bins) for the same data.
    Strong-signal objects (FAP≈0) almost always triggered the upgrade,
    causing MHAOV to find a sub-harmonic that MBLS did not, which sent
    30 correctly-detected objects to Tier 3 where they failed to publish.

    The nh_max, n_top_peaks, f_pval_thresh parameters are retained for
    API compatibility but have no effect.

    Returns
    -------
    (power_array, best_period, best_F, nh=2)
    """
    pow_best    = mhaov_periodogram(t, y, dy, test_periods, nh=nh_min)
    best_idx    = int(np.argmax(pow_best))
    best_period = float(test_periods[best_idx])
    best_F      = float(pow_best[best_idx])
    return pow_best, best_period, best_F, nh_min


# ── Conditional Entropy ───────────────────────────────────────────────────────

def ce_single(
    t:       np.ndarray,
    y:       np.ndarray,
    period:  float,
    n_phase: int = 10,
    n_mag:   int = 5,
) -> float:
    """CE score at a single trial period (lower = better phase coherence)."""
    phase  = (t / period) % 1.0
    p_bins = np.floor(phase * n_phase).astype(int) % n_phase
    m_min, m_max = y.min(), y.max()
    if m_max == m_min:
        return 1.0
    m_norm = (y - m_min) / (m_max - m_min)
    m_bins = np.floor(m_norm * n_mag).astype(int).clip(0, n_mag - 1)

    n  = len(t)
    ce = 0.0
    for pb in range(n_phase):
        mask = p_bins == pb
        n_p  = mask.sum()
        if n_p == 0:
            continue
        for mb in range(n_mag):
            n_pm = (mask & (m_bins == mb)).sum()
            if n_pm == 0:
                continue
            p_pm         = n_pm / n
            p_m_given_p  = n_pm / n_p
            ce          += p_pm * np.log(p_m_given_p)

    return -ce / (np.log(n_phase) + 1e-12)


def ce_periodogram(
    t:            np.ndarray,
    y:            np.ndarray,
    test_periods: np.ndarray,
    n_phase: int = 10,
    n_mag:   int = 5,
) -> np.ndarray:
    """CE score array over all trial periods."""
    return np.array([ce_single(t, y, p, n_phase, n_mag) for p in test_periods])


# ── Period agreement check ────────────────────────────────────────────────────

def check_agreement(
    p1:  float,
    p2:  float,
    p3:  float,
    tol: float = 0.10,
) -> Tuple[object, float]:
    """
    Check whether three period estimates agree within tolerance.
    Checks harmonic ratios ×1, ×0.5, ×2 for each pair.

    Returns
    -------
    (agrees, spread_pct)
        agrees     : True | "two_of_three" | False
        spread_pct : max fractional deviation from median
    """
    periods = np.array([p1, p2, p3])
    med     = np.median(periods)
    spread  = float(np.max(np.abs(periods - med) / (med + 1e-12)))

    def _close(a, b):
        return any(
            abs(a - b * m) / (b * m + 1e-12) <= tol
            for m in [1.0, 0.5, 2.0]
        )

    if _close(p1, p2) and _close(p2, p3) and _close(p1, p3):
        return True, spread
    if _close(p1, p2) or _close(p2, p3) or _close(p1, p3):
        return "two_of_three", spread
    return False, spread
