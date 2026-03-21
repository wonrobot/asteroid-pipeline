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

Window function
---------------
The window function is recomputed on the finer Tier 2 grid (the Tier 1
grid is too coarse to resolve alias structure at the precision needed
for consensus selection). Contamination scores for all three method peaks
are stored in Tier2Result and propagated to reliability.py.

When all three methods agree, the consensus period is chosen as the
least window-contaminated of the three estimates rather than a plain
median. When contamination is severe (>CONTAMINATION_THRESHOLD) for a
method's top peak, a warning is logged — the downstream reliability
code will lower the R-code accordingly.

MBLS false alarm probability (dual significance gate)
-----------------------------------------------------
MBLS operates on all photometric bands jointly — it has strictly more
information than MHAOV, which collapses multi-band data to a single
series. To use this information for the significance decision (not just
period estimation), we compute an empirical FAP for MBLS via permutation
test: shuffle time labels, recompute MBLS, repeat N times. The fraction
of permutations with max power >= observed power is the FAP.

Decision gate (dual):
  mhaov_sig = p_value   < cfg.tier.mhaov_pval_thresh
  mbls_sig  = mbls_fap  < cfg.tier.mbls_fap_thresh

  both_sig  → full confidence gate (supports R=3)
  either_sig → partial confidence gate (supports up to R=2)
  neither   → reject unless methods agree (escalate to Tier 3)

This means MBLS can now rescue a detection that MHAOV finds marginal
(common for faint multi-band objects with few obs per band), and
MHAOV can rescue a detection with unlucky permutation draws.

Permutation efficiency: uses a coarse 500-point grid for permutations
(null distribution needs max-power distribution, not resolved peaks).
Real MBLS runs on the full fine grid. Typical cost: +0.5–2s per asteroid.
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
    CONTAMINATION_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Minimum per-band chi-sq improvement to count a band as 'supporting' the period.
# 0.1 = weak but real signal in that band. Deliberately low — we want to know
# if the band has ANY preference for this period, not if it alone would detect it.
BAND_SUPPORT_THRESH = 0.10


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Tier2Result:
    """
    Output of Tier 2 period refinement.

    Attributes
    ----------
    provid              : asteroid designation
    passes              : True = all methods agree, proceed to publish
    to_tier3            : True = methods disagree, needs disambiguation
    best_period_mhaov   : MHAOV best period (hours)
    best_period_mbls    : MBLS Nterms=2 best period (hours)
    best_period_ce      : Conditional Entropy best period (hours)
    consensus_period    : best-period estimate from the agreeing methods
    F_stat              : MHAOV F-statistic at best period
    p_value             : MHAOV p-value at best period
    amplitude           : peak-to-peak of detrended lightcurve
    snr                 : signal-to-noise ratio
    agreement           : True / "two_of_three" / False
    period_spread_pct   : max fractional spread between the three periods
    reject_reason       : explanation if not passes and not to_tier3
    test_periods        : period grid
    mhaov_power         : MHAOV F-statistic array
    mbls_power          : MBLS power array
    ce_scores           : conditional entropy array (lower = better)
    window_power        : spectral window function on test_periods grid
    mhaov_contamination : cadence-alias score for MHAOV best period [0–1]
    mbls_contamination  : cadence-alias score for MBLS best period [0–1]
    ce_contamination    : cadence-alias score for CE best period [0–1]
    consensus_contamination : cadence-alias score for adopted consensus [0–1]
    mbls_fap                : MBLS false alarm probability via permutation [0–1]
    mbls_sig                : True if mbls_fap < cfg.tier.mbls_fap_thresh
    mhaov_sig               : True if p_value  < cfg.tier.mhaov_pval_thresh
    both_sig                : True if both significance gates passed
    mbls_band_support       : per-band chi-sq improvement at consensus period
    mbls_n_bands_supporting : number of bands with individual support score > threshold
    mbls_band_support_frac  : fraction of bands supporting (0–1)
    """
    provid:                  str
    passes:                  bool
    to_tier3:                bool
    best_period_mhaov:       float
    best_period_mbls:        float
    best_period_mbls_raw:    float  # before 2-minima doubling
    best_period_ce:          float
    consensus_period:        float
    F_stat:                  float
    p_value:                 float
    amplitude:               float
    snr:                     float
    agreement:               bool
    period_spread_pct:       float
    reject_reason:           Optional[str]
    test_periods:            np.ndarray
    mhaov_power:             np.ndarray
    mbls_power:              np.ndarray
    ce_scores:               np.ndarray
    # ── New: window function fields ───────────────────────────────────────────
    window_power:            np.ndarray   # window function on Tier 2 grid
    mhaov_contamination:     float        # cadence-alias score for MHAOV peak
    mbls_contamination:      float        # cadence-alias score for MBLS peak
    ce_contamination:        float        # cadence-alias score for CE peak
    consensus_contamination: float        # cadence-alias score for adopted period
    # ── New: MBLS false alarm probability ────────────────────────────────────
    mbls_fap:                float        # MBLS FAP via permutation test [0–1]
    mbls_sig:                bool         # True if mbls_fap < mbls_fap_thresh
    mhaov_sig:               bool         # True if p_value < mhaov_pval_thresh
    both_sig:                bool         # True if both gates passed
    # ── New: per-band MBLS support ────────────────────────────────────────────
    mbls_band_support:       dict         # {band: chi-sq improvement score}
    mbls_n_bands_supporting: int          # bands with score > BAND_SUPPORT_THRESH
    mbls_band_support_frac:  float        # fraction of bands supporting [0-1]


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
    t1_best    = t1result.best_period_mbls
    if t1_best < 0.5:
        p_min  = data.period_min_hr
        n_cap  = 50_000
    else:
        p_min  = max(data.period_min_hr, t1_best / 4.0)
        n_cap  = 8_000
    p_max = min(cfg_p.period_max_hr, data.baseline_hr)
    n_t2  = max(cfg_p.n_grid_fine,
                int(10 * data.baseline_hr * (1.0/p_min - 1.0/p_max)))
    n_t2  = min(n_t2, n_cap)
    test_periods = np.linspace(p_min, p_max, n_t2)

    # ── Window function on Tier 2 grid ────────────────────────────────────────
    # Recomputed on the finer grid — the Tier 1 grid is too coarse to
    # resolve alias structure at the precision needed for consensus selection.
    # This is the same data.t_hrs so the window shape is identical; only the
    # resolution improves. Cost is one GLS-on-ones call: negligible vs methods.
    window_pow = compute_window_function(data.t_hrs, test_periods)

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

    # Score MHAOV best period for window contamination
    mhaov_cont = contamination_score(best_mhaov, test_periods, window_pow)
    if mhaov_cont > CONTAMINATION_THRESHOLD:
        logger.warning(
            f"{data.provid}: MHAOV best={best_mhaov:.3f}hr is window-contaminated "
            f"(score={mhaov_cont:.2f}) — cadence alias risk"
        )

    # ── 2. MBLS Nterms=2 — raw multiband series, no pre-applied offsets ───────
    logger.debug(f"{data.provid}: running MBLS Nterms=2...")
    try:
        mbls_pow      = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods, nterms=cfg_p.mbls_nterms_t2
        )
        best_mbls_raw = test_periods[np.argmax(mbls_pow)]

        d_mhaov_mbls_raw    = abs(best_mhaov - best_mbls_raw) / max(best_mbls_raw, 1e-6)
        d_mhaov_mbls_double = abs(best_mhaov - 2*best_mbls_raw) / max(best_mbls_raw, 1e-6)

        apply_rule = (d_mhaov_mbls_raw    <= cfg_t.agreement_tol or
                      d_mhaov_mbls_double <= cfg_t.agreement_tol)

        if apply_rule:
            best_mbls, was_doubled, n_minima = apply_two_minima_rule(
                data.t_hrs, data.y_multiband, data.dy, data.bands,
                period=best_mbls_raw, nterms=cfg_p.mbls_nterms_t2,
            )
            logger.debug(
                f"{data.provid}: 2-minima rule applied "
                f"(MHAOV={best_mhaov:.3f}hr, MBLS={best_mbls_raw:.3f}hr "
                f"d={d_mhaov_mbls_raw:.1%} d2x={d_mhaov_mbls_double:.1%}) → "
                f"{'doubled to ' + str(round(best_mbls,3)) if was_doubled else 'kept at ' + str(round(best_mbls_raw,3))}"
            )
        else:
            best_mbls   = best_mbls_raw
            was_doubled = False
            n_minima    = -1
            logger.debug(
                f"{data.provid}: 2-minima rule skipped — "
                f"MHAOV={best_mhaov:.3f}hr vs MBLS={best_mbls_raw:.3f}hr "
                f"genuine disagreement (d={d_mhaov_mbls_raw:.1%}, "
                f"d2x={d_mhaov_mbls_double:.1%})"
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

    # Score MBLS best period
    mbls_cont = contamination_score(float(best_mbls), test_periods, window_pow)
    if mbls_cont > CONTAMINATION_THRESHOLD:
        logger.warning(
            f"{data.provid}: MBLS best={best_mbls:.3f}hr is window-contaminated "
            f"(score={mbls_cont:.2f})"
        )

    # ── MBLS false alarm probability — permutation test ───────────────────────
    # Uses a coarse 500-point grid for permutations: we only need the null
    # max-power distribution, not a resolved periodogram. The real MBLS peak
    # (from the fine grid above) is compared against this null distribution.
    # Coarse grid makes each permutation ~16× faster than the fine grid.
    observed_mbls_max = float(mbls_pow.max())
    mbls_fap = compute_mbls_fap(
        data.t_hrs, data.y_multiband, data.dy, data.bands,
        test_periods=test_periods,
        observed_max_power=observed_mbls_max,
        n_perm=cfg_t.mbls_fap_n_perm,
        nterms=cfg_p.mbls_nterms_t2,
    )
    mhaov_sig = p_value  < cfg_t.mhaov_pval_thresh
    mbls_sig  = mbls_fap < cfg_t.mbls_fap_thresh
    both_sig  = mhaov_sig and mbls_sig
    either_sig = mhaov_sig or mbls_sig

    logger.debug(
        f"{data.provid}: MBLS FAP={mbls_fap:.4f} "
        f"({'sig' if mbls_sig else 'not sig'})  "
        f"MHAOV p={p_value:.2e} "
        f"({'sig' if mhaov_sig else 'not sig'})  "
        f"both={both_sig}"
    )

    # ── 3. Conditional Entropy — skip below 1hr ───────────────────────────────
    CE_MIN_HR = 1.0
    ce_skip   = (data.period_min_hr < CE_MIN_HR)
    if ce_skip:
        logger.debug(
            f"{data.provid}: CE skipped "
            f"(Nyquist floor={data.period_min_hr*60:.1f}min < 60min)"
        )
        ce_periods = test_periods
        ce_scores  = np.ones(len(test_periods))
        best_ce    = best_mhaov
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

    # Score CE best period
    ce_cont = contamination_score(float(best_ce), test_periods, window_pow)
    if ce_cont > CONTAMINATION_THRESHOLD and not ce_skip:
        logger.warning(
            f"{data.provid}: CE best={best_ce:.3f}hr is window-contaminated "
            f"(score={ce_cont:.2f})"
        )

    # ── Agreement check ────────────────────────────────────────────────────────
    agrees, spread_pct = check_agreement(best_mhaov, best_mbls, best_ce, cfg_t.agreement_tol)

    # ── MBLS per-band support at consensus period ────────────────────────────
    # Fit MBLS at the current consensus period estimate and check whether each
    # band individually prefers this period over a flat (constant) model.
    # This is computed here using the pre-consensus estimate; it will be used
    # downstream in reliability.py to distinguish two_of_three cases where
    # multiple bands all agree (higher confidence) from cases where only one
    # band drives the MBLS result (lower confidence).
    band_support, n_bands_supporting, band_support_frac = compute_mbls_band_support(
        data.t_hrs, data.y_multiband, data.dy, data.bands,
        period=float(best_mbls),
        nterms=cfg_p.mbls_nterms_t2,
    )
    logger.debug(
        f"{data.provid}: MBLS band support at {best_mbls:.3f}hr: "
        f"{band_support} → {n_bands_supporting}/{len(band_support)} bands "
        f"(frac={band_support_frac:.2f})"
    )

    # ── Consensus period — prefer least window-contaminated estimate ──────────
    # When methods agree, instead of a plain median we weight toward the
    # estimate with lowest cadence-alias contamination. This prevents the
    # consensus being pulled toward a contaminated candidate when, e.g., one
    # method's peak sits exactly on a window spike but the others are clean.
    d_mhaov_half = abs(best_mbls - 2*best_mhaov) / (2*best_mhaov + 1e-12)
    d_ce_half    = abs(best_mbls - 2*best_ce)    / (2*best_ce    + 1e-12)
    mbls_doubled = d_mhaov_half <= cfg_t.agreement_tol or d_ce_half <= cfg_t.agreement_tol

    if agrees and mbls_doubled:
        consensus = float(best_mbls)
    else:
        consensus = _contamination_weighted_consensus(
            periods=[best_mhaov, best_mbls, best_ce],
            contaminations=[mhaov_cont, mbls_cont, ce_cont],
            fallback=float(np.median([best_mhaov, best_mbls, best_ce])),
        )

    consensus_cont = contamination_score(consensus, test_periods, window_pow)

    logger.debug(
        f"{data.provid} Tier2: MHAOV={best_mhaov:.3f}hr (cont={mhaov_cont:.2f})  "
        f"MBLS={best_mbls:.3f}hr (cont={mbls_cont:.2f})  "
        f"CE={best_ce:.3f}hr (cont={ce_cont:.2f})  "
        f"consensus={consensus:.3f}hr (cont={consensus_cont:.2f})  "
        f"agree={agrees}  F={F_best:.1f}  p={p_value:.2e}  "
        f"mbls_fap={mbls_fap:.4f}  both_sig={both_sig}"
    )

    # ── Decision ───────────────────────────────────────────────────────────────
    full_agreement = (agrees is True)
    two_of_three   = (agrees == "two_of_three")
    any_agreement  = full_agreement or two_of_three

    def _make_result(passes, to_tier3, agreement_val, reject_reason=None, consensus_p=None):
        cp = consensus_p if consensus_p is not None else consensus
        cp_cont = contamination_score(cp, test_periods, window_pow) if not np.isnan(cp) else np.nan
        return Tier2Result(
            provid=data.provid, passes=passes, to_tier3=to_tier3,
            best_period_mhaov=best_mhaov,
            best_period_mbls=best_mbls,
            best_period_mbls_raw=best_mbls_raw,
            best_period_ce=best_ce,
            consensus_period=cp,
            F_stat=F_best, p_value=p_value,
            amplitude=data.amplitude, snr=data.snr,
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
            consensus_contamination=cp_cont,
            mbls_fap=mbls_fap,
            mbls_sig=mbls_sig,
            mhaov_sig=mhaov_sig,
            both_sig=both_sig,
            mbls_band_support=band_support,
            mbls_n_bands_supporting=n_bands_supporting,
            mbls_band_support_frac=band_support_frac,
        )

    # ── Hard reject: neither gate significant and methods disagree ───────────
    # This is the only unconditional rejection path. If methods disagree AND
    # neither MHAOV nor MBLS finds the signal significant, there is nothing
    # to disambiguate — Tier 3 would be running CLEAN on noise.
    if not either_sig and not any_agreement:
        return _make_result(
            passes=False, to_tier3=False, agreement_val=False,
            reject_reason=(
                f"neither gate significant "
                f"(MHAOV p={p_value:.2e}, MBLS FAP={mbls_fap:.4f}) "
                f"and methods disagree"
            )
        )

    # Fix consensus for harmonic cases
    d_mbls_2mhaov_cons = abs(best_mbls - 2*best_mhaov) / (2*best_mhaov + 1e-12)
    if d_mbls_2mhaov_cons <= cfg_t.agreement_tol * 2:
        consensus = float(best_mbls)

    # ── Full agreement ─────────────────────────────────────────────────────────
    # Publish if either gate is significant.
    # both_sig → reliability.py will assign R=3 (high confidence)
    # either_sig only → reliability.py will cap at R=2 (moderate)
    if full_agreement and either_sig:
        return _make_result(passes=True, to_tier3=False, agreement_val=True)

    # ── Two-of-three agreement ─────────────────────────────────────────────────
    # Publish if either gate is significant.
    # Consensus period logic unchanged — which two methods agreed determines it.
    if two_of_three and either_sig:
        d_ce_mhaov    = abs(best_ce - best_mhaov) / (best_mhaov + 1e-12) if best_ce else 1.0
        d_mhaov_mbls  = abs(best_mhaov - best_mbls) / (best_mbls + 1e-12)
        d_mbls_2mhaov = abs(best_mbls - 2*best_mhaov) / (2*best_mhaov + 1e-12)
        if d_mbls_2mhaov <= cfg_t.agreement_tol * 2:
            consensus_two = best_mbls
            logger.debug(f"{data.provid} Tier2: 2-of-3 MBLS=2×MHAOV → consensus={consensus_two:.3f}hr")
        elif d_ce_mhaov <= cfg_t.agreement_tol * 2 and d_mhaov_mbls > cfg_t.agreement_tol:
            consensus_two = best_mhaov
            logger.debug(f"{data.provid} Tier2: 2-of-3 MHAOV+CE → consensus={consensus_two:.3f}hr")
        else:
            consensus_two = best_mbls
            logger.debug(f"{data.provid} Tier2: 2-of-3 MHAOV+MBLS → consensus={consensus_two:.3f}hr")
        return _make_result(
            passes=True, to_tier3=False, agreement_val="two_of_three",
            consensus_p=consensus_two
        )

    # ── Disagreement with partial significance → Tier 3 ─────────────────────
    # Methods disagree but at least one gate is significant — real signal
    # likely present but period is ambiguous. CLEAN can resolve it.
    return _make_result(passes=False, to_tier3=True, agreement_val=False)


# ── MBLS false alarm probability ─────────────────────────────────────────────

def compute_mbls_fap(
    t:                   np.ndarray,
    y:                   np.ndarray,
    dy:                  np.ndarray,
    bands:               np.ndarray,
    test_periods:        np.ndarray,
    observed_max_power:  float,
    n_perm:              int   = 200,
    nterms:              int   = 2,
    n_coarse:            int   = 500,
    seed:                int   = 42,
) -> float:
    """
    Estimate MBLS false alarm probability via permutation test.

    Shuffles the time labels n_perm times. For each shuffle, computes the
    maximum MBLS power on a coarse grid. The FAP is the fraction of
    permutations where max power >= observed_max_power.

    Why time-label shuffling?
    -------------------------
    Shuffling t while keeping y, dy, bands fixed destroys all temporal
    periodicity but preserves the noise properties, magnitude distribution,
    and band structure. This gives the correct null distribution for the
    hypothesis "there is no periodic signal in this data."

    Why a coarse grid for permutations?
    ------------------------------------
    The null distribution only needs the max-power statistic — we don't
    need to resolve individual peaks. A 500-point grid gives the same
    max-power distribution as an 8000-point grid but runs 16× faster.
    The fine-grid observed power is still used as the comparison value,
    so we're not losing precision on the detection side.

    Typical cost: ~0.5–2s per asteroid (200 perms × coarse grid).

    Parameters
    ----------
    test_periods         : fine-grid periods (used to set coarse grid range)
    observed_max_power   : max MBLS power on the fine grid (pre-computed)
    n_perm               : number of permutations (200 → FAP resolution 0.005)
    nterms               : MBLS Fourier terms (match what produced observed power)
    n_coarse             : grid points for permutation periodograms
    seed                 : random seed for reproducibility

    Returns
    -------
    fap : float in [0, 1]
        Fraction of permutations with max power >= observed.
        Low FAP (< 0.001) → signal is significant.
    """
    rng = np.random.default_rng(seed)

    # Coarse grid spanning the same range as the fine grid
    p_lo = float(test_periods[0])
    p_hi = float(test_periods[-1])
    coarse_periods = np.linspace(p_lo, p_hi, n_coarse)

    n_exceed = 0
    for _ in range(n_perm):
        t_perm = rng.permutation(t)
        try:
            pow_perm = mbls_periodogram(
                t_perm, y, dy, bands, coarse_periods, nterms=nterms
            )
            if float(pow_perm.max()) >= observed_max_power:
                n_exceed += 1
        except Exception:
            # If gatspy fails on a permutation (degenerate case), treat as
            # low power — do not count as exceeding observed.
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
    Compute per-band chi-sq improvement of MBLS fit at a given period.

    For each photometric band, measures how much better a Fourier model at
    `period` fits the data compared to a flat (constant) model. A band
    "supports" the period if its improvement exceeds BAND_SUPPORT_THRESH.

    This is the key quantity for Change 5: in the two_of_three reliability
    path (MHAOV+MBLS agree, CE disagrees), we check whether multiple bands
    independently agree on the period. If so, the CE disagreement is more
    likely to be a CE limitation (histogram under-sampling) than evidence
    against the period.

    Why chi-sq improvement per band?
    ---------------------------------
    MBLS maximises joint chi-sq across all bands simultaneously. The global
    result could be driven by one band with many observations while another
    band is essentially flat. Per-band chi-sq improvement isolates each
    band's individual contribution to the detection.

    Parameters
    ----------
    period  : period to evaluate (hours) — typically the MBLS consensus period
    nterms  : Fourier terms (match what was used for the MBLS periodogram)

    Returns
    -------
    band_support        : dict {band_name: chi_sq_improvement_fraction}
                          Values in [0, 1]: 0 = flat fits as well as periodic,
                          1 = periodic model explains all variance.
    n_bands_supporting  : int — number of bands with score > BAND_SUPPORT_THRESH
    band_support_frac   : float — n_bands_supporting / n_bands_total
    """
    from gatspy.periodic import LombScargleMultiband

    unique_bands = np.unique(bands)
    band_support = {}

    for band in unique_bands:
        mask = bands == band
        t_b  = t[mask]
        y_b  = y[mask]
        dy_b = dy[mask]

        if len(t_b) < 4:
            band_support[str(band)] = 0.0
            continue

        w     = 1.0 / dy_b**2
        y_wm  = float(np.average(y_b, weights=w))
        ss_tot = float(np.sum(w * (y_b - y_wm)**2))

        if ss_tot < 1e-12:
            band_support[str(band)] = 0.0
            continue

        # Fit Fourier model at this period for this band alone
        ph = 2.0 * np.pi * t_b / period
        cols = [np.ones(len(t_b))]
        for k in range(1, nterms + 1):
            cols += [np.cos(k * ph), np.sin(k * ph)]
        A = np.column_stack(cols)

        try:
            Aw     = A * w[:, None]
            coeffs = np.linalg.lstsq(Aw.T @ A, Aw.T @ y_b, rcond=None)[0]
            y_fit  = A @ coeffs
            ss_res = float(np.sum(w * (y_b - y_fit)**2))
            improvement = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
        except (np.linalg.LinAlgError, ValueError):
            improvement = 0.0

        band_support[str(band)] = improvement

    n_total     = len(unique_bands)
    n_supporting = sum(1 for v in band_support.values() if v >= BAND_SUPPORT_THRESH)
    frac         = float(n_supporting) / float(n_total) if n_total > 0 else 0.0

    return band_support, n_supporting, frac

# ── Contamination-weighted consensus ─────────────────────────────────────────

def _contamination_weighted_consensus(
    periods:        list,
    contaminations: list,
    fallback:       float,
) -> float:
    """
    Choose a consensus period that prefers less window-contaminated estimates.

    Strategy:
    - If one or more estimates has contamination < CONTAMINATION_THRESHOLD,
      take the median of only those clean estimates.
    - If all estimates are contaminated, fall back to plain median.
      (All three methods hitting window peaks simultaneously is itself
      informative — the reliability code will flag it.)
    """
    clean_mask = [c < CONTAMINATION_THRESHOLD for c in contaminations]
    if any(clean_mask):
        clean_periods = [p for p, m in zip(periods, clean_mask) if m]
        return float(np.median(clean_periods))
    return fallback


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
    Vectorised, chunked implementation.
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
    chunk  = 500

    for start in range(0, P, chunk):
        end  = min(start + chunk, P)
        tp   = test_periods[start:end]
        C    = len(tp)

        ph = 2.0 * np.pi * t[np.newaxis, :] / tp[:, np.newaxis]

        cols = [np.ones((C, N))]
        for k in range(1, nh + 1):
            cols.append(np.cos(k * ph))
            cols.append(np.sin(k * ph))
        A = np.stack(cols, axis=2)

        Aw = A * w[np.newaxis, :, np.newaxis]

        AtwA = np.einsum("cnk,cnl->ckl", Aw, A)
        Atwy = np.einsum("cnk,n->ck",    Aw, y)

        try:
            coeffs = np.linalg.solve(
                AtwA, Atwy[:, :, np.newaxis]
            )[:, :, 0]
        except (np.linalg.LinAlgError, ValueError):
            F_out[start:end] = np.array(
                [mhaov_single(t, y, dy, p, nh) for p in tp]
            )
            continue

        y_fit    = np.einsum("cnk,ck->cn", A, coeffs)
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
    Graham et al. (2013). Lower = better.
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
    """Conditional Entropy over a grid of trial periods. Lower = better."""
    N = len(t)
    P = len(test_periods)

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
        tp   = test_periods[start:end]
        C    = len(tp)

        phases     = (t[np.newaxis, :] % tp[:, np.newaxis]) / tp[:, np.newaxis]
        phase_bins = np.floor(phases * n_phase).astype(np.int32)
        phase_bins = np.clip(phase_bins, 0, n_phase - 1)

        lin_idx = phase_bins * n_mag + mag_bins[np.newaxis, :]

        for ci in range(C):
            counts      = np.bincount(lin_idx[ci], minlength=n_phase * n_mag)
            hist        = counts.reshape(n_phase, n_mag).astype(np.float64)
            phase_totals = hist.sum(axis=1)
            p_phase      = phase_totals / N

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
    p1 = MHAOV, p2 = MBLS (may be doubled by 2-minima rule), p3 = CE
    Returns (agrees, spread_pct).
    """
    tol_harmonic = tol * 2.0

    periods    = np.array([p1, p2, p3])
    median_p   = np.median(periods)
    spread_pct = float(np.max(np.abs(periods - median_p) / (median_p + 1e-12)))

    if spread_pct <= tol:
        return True, spread_pct

    d_mhaov_half = abs(p2 - 2*p1) / (2*p1 + 1e-12)
    d_ce_half    = abs(p2 - 2*p3) / (2*p3 + 1e-12)
    d_mbls_ce    = abs(p2 - p3)   / (p3   + 1e-12)
    d_mbls_mhaov = abs(p2 - p1)   / (p1   + 1e-12)

    if d_mhaov_half <= tol and d_ce_half <= tol_harmonic:
        return True, float(max(d_mhaov_half, d_ce_half))

    if d_mhaov_half <= tol and d_mbls_ce <= tol_harmonic:
        return True, float(max(d_mhaov_half, d_mbls_ce))

    if d_ce_half <= tol and d_mbls_mhaov <= tol_harmonic:
        return True, float(max(d_ce_half, d_mbls_mhaov))

    d_mhaov_double = abs(p1 - 2*p2) / (2*p2 + 1e-12)
    d_ce_double    = abs(p3 - 2*p2) / (2*p2 + 1e-12)
    d_ce_mhaov     = abs(p3 - p1)   / (p1   + 1e-12)

    if d_mhaov_double <= tol and d_ce_mhaov <= tol_harmonic:
        return True, float(max(d_mhaov_double, d_ce_mhaov))

    if d_mhaov_double <= tol and abs(p3-p2)/(p2+1e-12) <= tol_harmonic:
        return "two_of_three", float(d_mhaov_double)

    d_mhaov_mbls = abs(p1 - p2) / (p2 + 1e-12)
    if d_mhaov_mbls <= tol:
        return "two_of_three", float(d_mhaov_mbls)

    if d_ce_mhaov <= tol:
        return "two_of_three", float(d_ce_mhaov)

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
    Returns (corrected_period, was_doubled, n_minima).
    """
    from scipy.signal import find_peaks

    if dominant_band is None:
        unique, counts = np.unique(bands, return_counts=True)
        dominant_band  = unique[np.argmax(counts)]

    model = LombScargleMultiband(Nterms_base=nterms, Nterms_band=0)
    model.fit(t, y_multiband, dy, bands)

    phase_grid   = np.linspace(0, 1, n_phase)
    t_grid       = phase_grid * period
    fitted_curve = model.predict(
        t_grid,
        filts=np.full(n_phase, dominant_band),
        period=period,
    )

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
    """MHAOV at a single period — returns (F_stat, SS_resid, df_resid)."""
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
    """MHAOV at a single period with adaptive harmonic order selection."""
    from scipy.stats import f as f_dist

    F_best, SS_best, df_best = mhaov_single_sigma(t, y, dy, period, nh=nh_min)
    selected_nh  = nh_min
    was_upgraded = False

    for nh in range(nh_min + 1, nh_max + 1):
        N = len(t)
        if N - 2 * nh - 1 <= 0:
            break

        F_new, SS_new, df_new = mhaov_single_sigma(t, y, dy, period, nh=nh)

        delta_SS  = SS_best - SS_new
        extra_df  = 2
        if SS_new <= 0 or df_new <= 0:
            break

        F_nested = (delta_SS / extra_df) / (SS_new / df_new)
        p_nested = float(1.0 - f_dist.cdf(F_nested, extra_df, df_new))

        if p_nested < f_pval_thresh:
            F_best      = F_new
            SS_best     = SS_new
            df_best     = df_new
            selected_nh = nh
            was_upgraded = True
        else:
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
    """MHAOV periodogram with adaptive harmonic order selection."""
    from scipy.signal import find_peaks as _find_peaks

    base_power = mhaov_periodogram(t, y, dy, test_periods, nh=nh_min)

    peak_idxs, _ = _find_peaks(base_power, height=base_power.max() * 0.3)
    if len(peak_idxs) == 0:
        peak_idxs = np.array([np.argmax(base_power)])

    top_idxs = sorted(peak_idxs, key=lambda i: base_power[i], reverse=True)
    top_idxs = top_idxs[:n_top_peaks]

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
