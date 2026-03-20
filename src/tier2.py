"""
tier2.py
--------
Tier 2: Period refinement for objects that passed Tier 1.
Runs three independent methods and checks for agreement.

Methods
-------
1. MHAOV NH=2  — Multi-Harmonic AOV (Schwarzenberg-Czerny 1996)
                 Fits 2 harmonics, returns F-statistic with real p-values.

2. MBLS Nterms=2 — Multi-band LS with 2nd harmonic
                 Uses raw multiband series (no pre-applied band offsets).
                 MBLS fits per-band means internally.

3. Conditional Entropy — Graham et al. (2013)
                 Model-free validator. Lower = better.

Data routing
------------
MHAOV → data.y_dt        (merged, band-offset + detrended)
MBLS  → data.y_multiband (geometry-corrected only, raw band labels)
CE    → data.y_dt        (merged)
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
    reject_reason    : explanation if not passes and not to_tier3
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
    best_period_mbls_raw: float  # before 2-minima doubling
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
    """
    if not t1result.passes:
        raise ValueError(f"{data.provid}: Tier 1 did not pass — cannot run Tier 2")

    cfg_p = config.period
    cfg_t = config.tier

    # Fine period grid — dynamic sizing, data-driven floor
    p_min = data.period_min_hr
    p_max = min(cfg_p.period_max_hr, data.baseline_hr)
    # 10× oversampling for Tier 2 refinement (Greenstreet standard)
    n_t2  = max(cfg_p.n_grid_fine,
                int(10 * data.baseline_hr * (1.0/p_min - 1.0/p_max)))
    n_t2  = min(n_t2, 50_000)  # cap to keep runtime reasonable
    test_periods = np.linspace(p_min, p_max, n_t2)

    # ── 1. MHAOV adaptive NH — merged series ──────────────────────────────────
    logger.debug(f"{data.provid}: running MHAOV adaptive NH=2-4...")
    mhaov_pow, best_mhaov, F_best, best_nh = mhaov_periodogram_adaptive(
        data.t_hrs, data.y_dt, data.dy, test_periods,
        nh_min=cfg_p.mhaov_nh, nh_max=4, n_top_peaks=10, f_pval_thresh=0.10,
    )
    if best_nh > cfg_p.mhaov_nh:
        logger.debug(
            f"{data.provid}: MHAOV upgraded NH={cfg_p.mhaov_nh}→{best_nh} "
            f"at {best_mhaov:.3f}hr"
        )

    df_model = 2 * cfg_p.mhaov_nh
    df_resid = data.n_obs - 2 * cfg_p.mhaov_nh - 1
    p_value  = float(1.0 - f_dist.cdf(F_best, df_model, max(df_resid, 1)))

    # ── 2. MBLS Nterms=2 — raw multiband series, no pre-applied offsets ───────
    logger.debug(f"{data.provid}: running MBLS Nterms=2...")
    try:
        mbls_pow      = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods, nterms=cfg_p.mbls_nterms_t2
        )
        best_mbls_raw = test_periods[np.argmax(mbls_pow)]

        # Apply 2-minima rule (Greenstreet et al. 2026)
        best_mbls, was_doubled, n_minima = apply_two_minima_rule(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            period=best_mbls_raw, nterms=cfg_p.mbls_nterms_t2,
        )
        if was_doubled:
            logger.debug(
                f"{data.provid}: 2-minima rule: "
                f"{best_mbls_raw:.3f}hr ({n_minima} minima) "
                f"→ doubled to {best_mbls:.3f}hr"
            )
    except Exception as e:
        logger.warning(f"{data.provid}: MBLS Tier2 failed ({e}) — using MHAOV period")
        mbls_pow      = mhaov_pow.copy()
        best_mbls_raw = best_mhaov
        best_mbls     = best_mhaov

    # ── 3. Conditional Entropy — skip below 1hr (histogram unreliable) ────────
    # CE is unreliable below ~1hr because the phase histogram bins are too
    # sparse to resolve fast rotator peaks. Skip CE for fast rotators and use
    # MHAOV+MBLS two-of-three agreement instead (Greenstreet-equivalent).
    CE_MIN_HR = 1.0
    ce_skip   = (data.period_min_hr < CE_MIN_HR)
    if ce_skip:
        logger.debug(
            f"{data.provid}: CE skipped "
            f"(Nyquist floor={data.period_min_hr*60:.1f}min < 60min)"
        )
        ce_periods = test_periods
        ce_scores  = np.ones(len(test_periods))  # flat — no CE signal
        best_ce    = best_mhaov  # forces two_of_three path
    else:
        logger.debug(f"{data.provid}: running Conditional Entropy...")
        ce_periods = np.linspace(
            max(data.period_min_hr, CE_MIN_HR),
            min(cfg_p.period_max_hr, data.baseline_hr),
            cfg_p.n_grid_ce,
        )
        ce_scores = ce_periodogram(
            data.t_hrs, data.y_dt, ce_periods,
            n_phase=cfg_p.ce_n_phase, n_mag=cfg_p.ce_n_mag
        )
        best_ce = ce_periods[np.argmin(ce_scores)]

    # ── Agreement check ────────────────────────────────────────────────────────
    agrees, spread_pct = check_agreement(best_mhaov, best_mbls, best_ce, cfg_t.agreement_tol)

    # Consensus: if MBLS doubled (harmonic case), weight toward MBLS
    d_mhaov_half = abs(best_mbls - 2*best_mhaov) / (2*best_mhaov + 1e-12)
    d_ce_half    = abs(best_mbls - 2*best_ce)    / (2*best_ce    + 1e-12)
    mbls_doubled = d_mhaov_half <= cfg_t.agreement_tol or d_ce_half <= cfg_t.agreement_tol

    if agrees and mbls_doubled:
        consensus = float(best_mbls)
    else:
        consensus = float(np.median([best_mhaov, best_mbls, best_ce]))

    logger.debug(
        f"{data.provid} Tier2: MHAOV={best_mhaov:.3f}hr  "
        f"MBLS={best_mbls:.3f}hr  CE={best_ce:.3f}hr  "
        f"agree={agrees}  F={F_best:.1f}  p={p_value:.2e}"
    )

    # ── Decision ───────────────────────────────────────────────────────────────
    full_agreement = (agrees is True)
    two_of_three   = (agrees == "two_of_three")
    any_agreement  = full_agreement or two_of_three

    # Helper to build Tier2Result — avoids repeating all fields
    def _make_result(passes, to_tier3, agreement_val, reject_reason=None, consensus_p=None):
        return Tier2Result(
            provid=data.provid, passes=passes, to_tier3=to_tier3,
            best_period_mhaov=best_mhaov,
            best_period_mbls=best_mbls,
            best_period_mbls_raw=best_mbls_raw,
            best_period_ce=best_ce,
            consensus_period=consensus_p if consensus_p is not None else consensus,
            F_stat=F_best, p_value=p_value,
            amplitude=data.amplitude, snr=data.snr,
            agreement=agreement_val,
            period_spread_pct=spread_pct,
            reject_reason=reject_reason,
            test_periods=test_periods,
            mhaov_power=mhaov_pow,
            mbls_power=mbls_pow,
            ce_scores=ce_scores,
        )

    if p_value >= cfg_t.mhaov_pval_thresh and not any_agreement:
        return _make_result(
            passes=False, to_tier3=False, agreement_val=False,
            reject_reason=f"p-value={p_value:.2e} not significant and methods disagree"
        )

    if full_agreement and p_value < cfg_t.mhaov_pval_thresh:
        return _make_result(passes=True, to_tier3=False, agreement_val=True)

    if two_of_three and p_value < cfg_t.mhaov_pval_thresh:
        logger.debug(f"{data.provid} Tier2: 2-of-3 agreement (MHAOV+MBLS) — tentative publish")
        return _make_result(
            passes=True, to_tier3=False, agreement_val="two_of_three",
            consensus_p=best_mbls
        )

    # Full disagreement — escalate to Tier 3
    return _make_result(passes=False, to_tier3=True, agreement_val=False)



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
    Schwarzenberg-Czerny (1996).
    """
    w  = 1.0 / dy**2
    N  = len(t)
    ph = 2.0 * np.pi * t / period

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
    """
    MHAOV F-statistic over a grid of trial periods.

    Vectorised implementation: builds all design matrices simultaneously
    using numpy broadcasting and solves all P linear systems in one batch.
    ~10x faster than the Python loop version.

    Processes in chunks of 500 periods to limit memory usage to ~50MB.
    """
    N        = len(t)
    P        = len(test_periods)
    w        = 1.0 / dy**2
    y_wmean  = float(np.average(y, weights=w))
    df_model = 2 * nh
    df_resid = N - 2 * nh - 1

    if df_resid <= 0:
        return np.zeros(P)

    F_out  = np.zeros(P)
    chunk  = 500   # process 500 periods at a time to limit memory

    for start in range(0, P, chunk):
        end  = min(start + chunk, P)
        tp   = test_periods[start:end]   # (C,)
        C    = len(tp)

        # Phase matrix: (C, N)
        ph = 2.0 * np.pi * t[np.newaxis, :] / tp[:, np.newaxis]

        # Design matrix: (C, N, 1+2*nh)
        cols = [np.ones((C, N))]
        for k in range(1, nh + 1):
            cols.append(np.cos(k * ph))
            cols.append(np.sin(k * ph))
        A = np.stack(cols, axis=2)   # (C, N, 1+2*nh)

        # Weighted: Aw[c,n,k] = A[c,n,k] * w[n]
        Aw = A * w[np.newaxis, :, np.newaxis]   # (C, N, 1+2*nh)

        # Normal equations: (C, 1+2*nh, 1+2*nh) and (C, 1+2*nh)
        AtwA = np.einsum("cnk,cnl->ckl", Aw, A)
        Atwy = np.einsum("cnk,n->ck",    Aw, y)

        try:
            # np.linalg.solve batch needs RHS shape (C, m, 1)
            coeffs = np.linalg.solve(
                AtwA, Atwy[:, :, np.newaxis]
            )[:, :, 0]   # (C, 1+2*nh)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to per-period solve for this chunk
            F_out[start:end] = np.array(
                [mhaov_single(t, y, dy, p, nh) for p in tp]
            )
            continue

        y_fit    = np.einsum("cnk,ck->cn", A, coeffs)     # (C, N)
        SS_model = np.sum(w[np.newaxis, :] * (y_fit - y_wmean)**2, axis=1)
        SS_resid = np.sum(w[np.newaxis, :] * (y[np.newaxis, :] - y_fit)**2, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(
                SS_resid > 0,
                (SS_model / df_model) / (SS_resid / df_resid),
                0.0,
            )
        F_out[start:end] = np.clip(F, 0.0, None)

    return F_out


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
    Graham et al. (2013). Lower = better (true period → structured lightcurve).
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
    chunk:        int = 500,
) -> np.ndarray:
    """
    Conditional Entropy over a grid of trial periods. Lower = better.

    Vectorised: pre-computes phase bins for all periods simultaneously
    using numpy broadcasting, then uses np.bincount per period.
    ~4x faster than the Python loop version.

    Parameters
    ----------
    chunk : number of periods to process at once (memory vs speed trade-off)
    """
    N = len(t)
    P = len(test_periods)

    # Digitize magnitudes once — same for all periods
    y_min, y_max = float(y.min()), float(y.max())
    if y_max == y_min:
        return np.ones(P)
    mag_bins = np.floor(
        (y - y_min) / (y_max - y_min + 1e-10) * n_mag
    ).astype(np.int32)
    mag_bins = np.clip(mag_bins, 0, n_mag - 1)

    ce_out = np.empty(P)

    for start in range(0, P, chunk):
        end  = min(start + chunk, P)
        tp   = test_periods[start:end]   # (C,)
        C    = len(tp)

        # Phase bins for all C periods: (C, N)
        phases     = (t[np.newaxis, :] % tp[:, np.newaxis]) / tp[:, np.newaxis]
        phase_bins = np.floor(phases * n_phase).astype(np.int32)
        phase_bins = np.clip(phase_bins, 0, n_phase - 1)

        # Linear index into flattened n_phase × n_mag grid: (C, N)
        lin_idx = phase_bins * n_mag + mag_bins[np.newaxis, :]

        for ci in range(C):
            counts      = np.bincount(lin_idx[ci], minlength=n_phase * n_mag)
            hist        = counts.reshape(n_phase, n_mag).astype(np.float64)
            phase_totals = hist.sum(axis=1)               # (n_phase,)
            p_phase      = phase_totals / N               # (n_phase,)

            # Conditional entropy H(mag|phase) = -Σ p(φ) Σ p(m|φ) log p(m|φ)
            ce = 0.0
            for pi in range(n_phase):
                if phase_totals[pi] == 0:
                    continue
                p_m_given_ph = hist[pi] / phase_totals[pi]
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_p = np.where(p_m_given_ph > 0,
                                     np.log(p_m_given_ph), 0.0)
                ce -= float(p_phase[pi] * np.dot(p_m_given_ph, log_p))
            ce_out[start + ci] = ce

    return ce_out


# ── Agreement check ───────────────────────────────────────────────────────────

def check_agreement(
    p1:  float,
    p2:  float,
    p3:  float,
    tol: float = 0.05,
) -> Tuple[object, float]:
    """
    Check whether period estimates agree within fractional tolerance.

    Standard check: all 3 within tol → agree.

    Harmonic cases (A, B, C): MBLS (p2) applied the 2-minima rule and
    doubled a P/2 alias. These use tol_harmonic=2*tol for CE comparisons
    because CE is a histogram method and inherently noisier than MHAOV/MBLS
    near alias peaks. Greenstreet et al. (2026) use 10% agreement tolerance
    between their two methods; our harmonic cases use the same threshold.

    p1 = MHAOV, p2 = MBLS (may be doubled by 2-minima rule), p3 = CE
    Returns (agrees, spread_pct).
    """
    tol_harmonic = tol * 2.0   # 10% for CE in harmonic cases (Greenstreet 2026)

    # Standard check: all 3 within tol
    periods    = np.array([p1, p2, p3])
    median_p   = np.median(periods)
    spread_pct = float(np.max(np.abs(periods - median_p) / (median_p + 1e-12)))

    if spread_pct <= tol:
        return True, spread_pct

    # Harmonic Cases A, B, C:
    # MBLS doubled a P/2 alias — check harmonic relationships
    d_mhaov_half = abs(p2 - 2*p1) / (2*p1 + 1e-12)
    d_ce_half    = abs(p2 - 2*p3) / (2*p3 + 1e-12)
    d_mbls_ce    = abs(p2 - p3)   / (p3   + 1e-12)
    d_mbls_mhaov = abs(p2 - p1)   / (p1   + 1e-12)

    # Case A: MBLS = 2*MHAOV AND MBLS = 2*CE (both others on P/2)
    if d_mhaov_half <= tol and d_ce_half <= tol_harmonic:
        return True, float(max(d_mhaov_half, d_ce_half))

    # Case B: MBLS = 2*MHAOV AND MBLS ≈ CE (CE near doubled period)
    if d_mhaov_half <= tol and d_mbls_ce <= tol_harmonic:
        return True, float(max(d_mhaov_half, d_mbls_ce))

    # Case C: MBLS = 2*CE AND MBLS ≈ MHAOV
    if d_ce_half <= tol and d_mbls_mhaov <= tol_harmonic:
        return True, float(max(d_ce_half, d_mbls_mhaov))

    # Two-of-three check: MHAOV (p1) and MBLS (p2) agree, CE (p3) does not
    # Equivalent to Greenstreet LSM+Fourier agreement criterion
    d_mhaov_mbls = abs(p1 - p2) / (p2 + 1e-12)
    if d_mhaov_mbls <= tol:
        return "two_of_three", float(d_mhaov_mbls)

    return False, spread_pct


def apply_two_minima_rule(
    t:             np.ndarray,
    y_multiband:   np.ndarray,
    dy:            np.ndarray,
    bands:         np.ndarray,
    period:        float,
    nterms:        int   = 2,
    n_phase:       int   = 500,
    prominence:    float = 0.01,
    dominant_band: str   = None,
) -> tuple:
    """
    Apply the 2-minima rule to an MBLS period candidate.

    An elongated asteroid produces TWO brightness minima per rotation.
    If the fitted lightcurve shows fewer than 2 minima, the period is
    likely P/2 and is doubled. Follows Greenstreet et al. (2026) Sec 3.2.

    Returns (corrected_period, was_doubled, n_minima)
    """
    from scipy.signal import find_peaks

    # Determine dominant band
    if dominant_band is None:
        unique, counts = np.unique(bands, return_counts=True)
        dominant_band  = unique[np.argmax(counts)]

    # Fit MBLS model at this period
    model = LombScargleMultiband(Nterms_base=nterms, Nterms_band=0)
    model.fit(t, y_multiband, dy, bands)

    # Evaluate on dense phase grid
    phase_grid   = np.linspace(0, 1, n_phase)
    t_grid       = phase_grid * period
    fitted_curve = model.predict(
        t_grid,
        filts=np.full(n_phase, dominant_band),
        period=period,
    )

    # Count minima with wraparound handling — tile the curve twice,
    # find all minima, keep only those in the first copy (phase 0-1)
    tiled      = np.concatenate([fitted_curve, fitted_curve])
    all_mins, _= find_peaks(-tiled, prominence=prominence)
    n_minima   = int(np.sum(all_mins < n_phase))

    if n_minima < 2:
        return period * 2.0, True, n_minima
    else:
        return period, False, n_minima


def mhaov_single_sigma(
    t:      np.ndarray,
    y:      np.ndarray,
    dy:     np.ndarray,
    period: float,
    nh:     int = 2,
) -> tuple:
    """
    MHAOV at a single period — returns (F_stat, SS_resid, df_resid).
    Extended version of mhaov_single that also returns residuals for
    F-test comparison between nested models.
    """
    from scipy.stats import f as f_dist
    w  = 1.0 / dy**2
    N  = len(t)
    ph = 2.0 * np.pi * t / period

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
        SS_model = np.sum(w * (y_fit - y_wmean)**2)
        SS_resid = np.sum(w * (y     - y_fit  )**2)
        df_model = 2 * nh
        df_resid = N - 2 * nh - 1
        if df_resid <= 0 or SS_resid == 0:
            return 0.0, np.inf, 1
        F_stat = (SS_model / df_model) / (SS_resid / df_resid)
        return float(F_stat), float(SS_resid), int(df_resid)
    except np.linalg.LinAlgError:
        return 0.0, np.inf, 1


def mhaov_adaptive_period(
    t:           np.ndarray,
    y:           np.ndarray,
    dy:          np.ndarray,
    period:      float,
    nh_min:      int   = 2,
    nh_max:      int   = 4,
    f_pval_thresh: float = 0.10,
) -> tuple:
    """
    Test MHAOV at a single period with adaptive harmonic order selection.

    Starts at nh_min (=2), tests whether nh+1 is significantly better
    via F-test. Accepts higher order only if p < f_pval_thresh.

    This follows the Greenstreet / Vavilov & Carry approach of selecting
    the simplest model not significantly worse than a more complex one.

    Parameters
    ----------
    f_pval_thresh : p-value threshold for accepting a higher NH
                    0.10 = accept if 90% confident higher order helps

    Returns
    -------
    (F_stat, selected_nh, was_upgraded)
    """
    from scipy.stats import f as f_dist

    F_best, SS_best, df_best = mhaov_single_sigma(t, y, dy, period, nh=nh_min)
    selected_nh  = nh_min
    was_upgraded = False

    for nh in range(nh_min + 1, nh_max + 1):
        # Check we have enough degrees of freedom
        N = len(t)
        if N - 2 * nh - 1 <= 0:
            break

        F_new, SS_new, df_new = mhaov_single_sigma(t, y, dy, period, nh=nh)

        # F-test: is NH=nh significantly better than NH=nh-1?
        # H0: the extra harmonics explain no additional variance
        # Extra params: 2 (one cos + one sin term added)
        delta_SS  = SS_best - SS_new        # reduction in residual SS
        extra_df  = 2                        # 2 extra parameters
        if SS_new <= 0 or df_new <= 0:
            break

        F_nested = (delta_SS / extra_df) / (SS_new / df_new)
        p_nested = float(1.0 - f_dist.cdf(F_nested, extra_df, df_new))

        if p_nested < f_pval_thresh:
            # Higher order significantly better — upgrade
            F_best      = F_new
            SS_best     = SS_new
            df_best     = df_new
            selected_nh = nh
            was_upgraded = True
        else:
            # No significant improvement — stop
            break

    return F_best, selected_nh, was_upgraded


def mhaov_periodogram_adaptive(
    t:             np.ndarray,
    y:             np.ndarray,
    dy:            np.ndarray,
    test_periods:  np.ndarray,
    nh_min:        int   = 2,
    nh_max:        int   = 4,
    n_top_peaks:   int   = 10,
    f_pval_thresh: float = 0.10,
) -> tuple:
    """
    MHAOV periodogram with adaptive harmonic order selection.

    Efficient two-step approach:
    1. Run NH=nh_min across full period grid (fast)
    2. Find top n_top_peaks candidate periods
    3. At each candidate, test NH up to nh_max via F-test
    4. Return full periodogram (from step 1) and best adaptive period

    Parameters
    ----------
    n_top_peaks : number of candidate peaks to test with adaptive NH
    f_pval_thresh : p-value threshold for accepting higher NH (0.10 typical)

    Returns
    -------
    (base_power, best_period, best_F, best_nh)
    base_power  : NH=nh_min power array (full grid, for plotting)
    best_period : period with highest adaptive F-stat
    best_F      : F-stat at best period
    best_nh     : NH selected at best period
    """
    from scipy.signal import find_peaks as _find_peaks

    # Step 1: Full grid scan at NH=nh_min
    base_power = mhaov_periodogram(t, y, dy, test_periods, nh=nh_min)

    # Step 2: Find top candidate peaks
    peak_idxs, _ = _find_peaks(base_power, height=base_power.max() * 0.3)
    if len(peak_idxs) == 0:
        peak_idxs = np.array([np.argmax(base_power)])

    top_idxs = sorted(peak_idxs, key=lambda i: base_power[i], reverse=True)
    top_idxs = top_idxs[:n_top_peaks]

    # Step 3: Adaptive NH at each candidate
    best_F      = -np.inf
    best_period = test_periods[np.argmax(base_power)]
    best_nh     = nh_min

    for idx in top_idxs:
        p = test_periods[idx]
        F_adapt, nh_adapt, upgraded = mhaov_adaptive_period(
            t, y, dy, p,
            nh_min=nh_min, nh_max=nh_max,
            f_pval_thresh=f_pval_thresh,
        )
        if F_adapt > best_F:
            best_F      = F_adapt
            best_period = p
            best_nh     = nh_adapt

    return base_power, best_period, float(best_F), best_nh
