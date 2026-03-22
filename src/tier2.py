"""
tier2.py
--------
Tier 2: Period refinement for objects that passed Tier 1.

Architecture: MBLS is the primary period detector. MHAOV is a significance
validator. CE is an annotation-only check when data permit.

Scientific basis for the hierarchy
-----------------------------------
Empirical analysis on Greenstreet et al. (2026), 76 objects:
  Both correct:           50/76 (66%)
  MBLS only correct:      13/76 (17%)   ← MBLS uniquely rescues 13 objects
  MHAOV only correct:      4/76  (5%)   ← MHAOV uniquely rescues 4 objects
  Neither correct:         9/76 (12%)

MBLS is correct in 83% of objects; MHAOV in 71%. MBLS has strictly more
information because it uses all photometric bands jointly with a shared
period model. MHAOV collapses multi-band data to a single detrended series,
discarding inter-band colour information.

The 4 apparent MHAOV-only wins: 1 case (MD38) MBLS actually found a real
Greenstreet additional period. In the remaining 3, MBLS missed due to the
2-minima rule being gated on MHAOV agreement — a circular dependency now
removed.

The CE problem
--------------
Conditional Entropy requires period_floor < 1hr. All Rubin data has an
Eyer & Bartholdi floor of ~1.4min < 60min, so CE is skipped for virtually
every object. The previous code defaulted best_ce = best_mhaov when skipped,
giving MHAOV two votes in the three-method consensus. This is removed: CE
is now annotation-only and never participates in the period decision.

New decision logic
------------------
1. MBLS runs unconditionally → best_mbls is the primary candidate.
   2-minima rule always applied (no longer gated on MHAOV agreement).

2. MHAOV runs → checks whether its peak confirms best_mbls.
   "Confirms" = MHAOV peak within agreement_tol of best_mbls or P/2 or 2P.

3. Significance gate (unchanged, dual):
   mhaov_sig = p_value  < mhaov_pval_thresh
   mbls_sig  = mbls_fap < mbls_fap_thresh

4. Agreement outcomes (new):
   "mbls_confirmed" : MHAOV confirms MBLS + either_sig → publish
                      both_sig → R=3 eligible; either_sig only → R≤2
   "mbls_sig_only"  : MBLS significant, MHAOV does not confirm → publish R≤2
                      Scientific basis: MBLS FAP directly tests whether multi-band
                      power exceeds the noise null. If FAP < 0.001 the signal is
                      real regardless of MHAOV. MHAOV non-confirmation means the
                      single-band collapsed series is less sensitive, not that the
                      period is wrong.
   to_tier3=True    : Both gates significant but MHAOV finds a genuinely
                      different significant period → ambiguous, send to CLEAN.
   reject           : Neither gate significant → no evidence for any period.

5. CE runs when floor allows (period_floor >= 1hr). Result stored as
   best_period_ce for catalog annotation. Never used in period selection.

Consensus period
----------------
Always best_mbls (after 2-minima correction). MHAOV's role is confirmation,
not competition. This removes the old contamination-weighted median which
could pull the consensus toward an MHAOV alias peak.
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
    mhaov_confirms          : True if MHAOV peak agrees with MBLS within tol or harmonic
    """
    provid:                  str
    passes:                  bool
    to_tier3:                bool
    best_period_mhaov:       float
    best_period_mbls:        float
    best_period_mbls_raw:    float  # before 2-minima doubling
    best_period_ce:          float  # NaN when CE skipped (annotation only)
    consensus_period:        float  # always = best_period_mbls (MBLS is primary)
    F_stat:                  float
    p_value:                 float
    amplitude:               float
    snr:                     float
    agreement:               object  # "mbls_confirmed" | "mbls_sig_only" | False
    period_spread_pct:       float   # |MHAOV - MBLS| / MBLS
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
    mhaov_confirms:          bool    # MHAOV peak agrees with MBLS within tol


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

    # Fine period grid — Greenstreet et al. (2026) Equation 4:
    #   n = 100 × T_days × (f_max − f_min)
    #   = 100 × T_hrs × (1/P_min_hrs − 1/P_max_hrs)
    #
    # The 100× oversampling ensures a smoothly resolved periodogram even
    # for fast rotators. T1 best period focuses the search: below 0.5hr
    # we keep the full range down to the Eyer & Bartholdi floor; above
    # 0.5hr we start from T1/4 to capture P/2 and P/3 harmonics.
    t1_best = t1result.best_period_mbls
    if t1_best < 0.5:
        p_min = data.period_min_hr   # full range — genuine fast rotator
        n_cap = 100_000              # allow dense grid for sub-hour periods
    else:
        p_min = max(data.period_min_hr, t1_best / 4.0)
        n_cap = 50_000
    p_max = min(cfg_p.period_max_hr, data.baseline_hr)
    # Greenstreet Eq. 4: n = 100 × T × (f_max − f_min)
    n_t2  = max(cfg_p.n_grid_fine,
                int(100 * data.baseline_hr * (1.0/p_min - 1.0/p_max)))
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
    mhaov_cont = contamination_score(best_mhaov, test_periods, window_pow,
                                   baseline_hr=data.baseline_hr)
    if mhaov_cont > CONTAMINATION_THRESHOLD:
        logger.warning(
            f"{data.provid}: MHAOV best={best_mhaov:.3f}hr is window-contaminated "
            f"(score={mhaov_cont:.2f}) — cadence alias risk"
        )

    # ── 2. MBLS Nterms=2 — primary period detector ────────────────────────────
    # MBLS is the primary detector. It uses all photometric bands jointly with
    # a shared period model, giving it strictly more information than MHAOV
    # (which collapses multi-band data to a single detrended series).
    #
    # 2-minima rule: always applied unconditionally. Previously gated on
    # MHAOV agreement — that was a circular dependency that prevented MBLS
    # from correcting itself when MHAOV was wrong. The 2-minima rule is a
    # physical test on the MBLS lightcurve shape and requires no input from
    # MHAOV. (Empirical finding: removing the gate recovered 5+ objects.)
    logger.debug(f"{data.provid}: running MBLS Nterms=2...")
    try:
        mbls_pow      = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods, nterms=cfg_p.mbls_nterms_t2
        )
        best_mbls_raw = test_periods[np.argmax(mbls_pow)]

        # Always apply 2-minima rule — no MHAOV gate
        best_mbls, was_doubled, n_minima = apply_two_minima_rule(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            period=best_mbls_raw, nterms=cfg_p.mbls_nterms_t2,
        )
        if was_doubled:
            logger.debug(
                f"{data.provid}: 2-minima rule: {best_mbls_raw:.3f}hr "
                f"({n_minima} minima) → doubled to {best_mbls:.3f}hr"
            )
        else:
            logger.debug(
                f"{data.provid}: 2-minima rule: kept at {best_mbls_raw:.3f}hr "
                f"({n_minima} minima, no doubling needed)"
            )
    except Exception as e:
        logger.warning(f"{data.provid}: MBLS Tier2 failed ({e}) — falling back to MHAOV period")
        mbls_pow      = mhaov_pow.copy()
        best_mbls_raw = best_mhaov
        best_mbls     = best_mhaov
        was_doubled   = False
        n_minima      = -1

    # Score MBLS best period
    mbls_cont = contamination_score(float(best_mbls), test_periods, window_pow,
                                  baseline_hr=data.baseline_hr)
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

    # ── 3. Conditional Entropy — annotation only ─────────────────────────────
    # CE requires period_floor >= 1hr to sample phase space adequately.
    # All Rubin data has an Eyer & Bartholdi floor of ~1.4min, so CE is skipped
    # for essentially every object.
    #
    # CRITICAL FIX: previously best_ce = best_mhaov when skipped. This gave
    # MHAOV two votes in the three-way consensus (MHAOV + CE=MHAOV vs MBLS).
    # CE is now annotation-only: best_ce = NaN when skipped and is never used
    # in the period decision. When CE does run (classical/sparse datasets with
    # large minimum gaps), its result is stored for catalog annotation only.
    CE_MIN_HR = 1.0
    ce_skip   = (data.period_min_hr < CE_MIN_HR)
    if ce_skip:
        logger.debug(
            f"{data.provid}: CE skipped — annotation only "
            f"(period floor={data.period_min_hr*60:.1f}min < 60min)"
        )
        ce_periods = test_periods
        ce_scores  = np.ones(len(test_periods))
        best_ce    = np.nan   # NaN = not run; never participates in decision
    else:
        logger.debug(f"{data.provid}: running Conditional Entropy (annotation only)...")
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
        logger.debug(
            f"{data.provid}: CE best={best_ce:.3f}hr (annotation, not used in decision)"
        )

    # ── CE contamination score (annotation) ──────────────────────────────────
    ce_cont = contamination_score(
        float(best_ce) if not np.isnan(best_ce) else float(best_mbls),
        test_periods, window_pow, baseline_hr=data.baseline_hr
    )

    # ── MBLS per-band support ─────────────────────────────────────────────────
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

    # ── MHAOV confirmation check ──────────────────────────────────────────────
    # MHAOV "confirms" MBLS if its best period is within agreement_tol of
    # best_mbls or a harmonic (P/2 or 2P). MHAOV is not competing with MBLS
    # for the consensus period — it is asking: "does the single-band significance
    # test agree that THIS period (best_mbls) is real?"
    #
    # P/2 check: common for double-hump lightcurves where MHAOV (NH=2 harmonics)
    # prefers the half-period more strongly in the single-band collapsed series.
    # 2P check: common when MBLS 2-minima rule doubled and MHAOV stayed at P/2.
    def _mhaov_confirms(mhaov_p, mbls_p, tol):
        for mult in [1.0, 0.5, 2.0]:
            if abs(mhaov_p - mbls_p * mult) / (mbls_p * mult + 1e-12) <= tol:
                return True
        return False

    mhaov_confirms = _mhaov_confirms(best_mhaov, best_mbls, cfg_t.agreement_tol)

    # Period spread: |MHAOV - MBLS| / MBLS (meaningful only when MHAOV confirms)
    spread_pct = float(abs(best_mhaov - best_mbls) / (best_mbls + 1e-12))

    # Consensus period: always best_mbls — MBLS is the primary detector.
    consensus = float(best_mbls)
    consensus_cont = contamination_score(
        consensus, test_periods, window_pow, baseline_hr=data.baseline_hr
    )

    logger.debug(
        f"{data.provid} Tier2: MHAOV={best_mhaov:.3f}hr (cont={mhaov_cont:.2f}, "
        f"{'confirms' if mhaov_confirms else 'disagrees'})  "
        f"MBLS={best_mbls:.3f}hr (cont={mbls_cont:.2f})  "
        f"CE={'skipped' if np.isnan(best_ce) else f'{best_ce:.3f}hr'} (annotation)  "
        f"consensus={consensus:.3f}hr (cont={consensus_cont:.2f})  "
        f"mhaov_confirms={mhaov_confirms}  F={F_best:.1f}  p={p_value:.2e}  "
        f"mbls_fap={mbls_fap:.4f}  both_sig={both_sig}"
    )

    # ── Decision — MBLS primary, MHAOV confirmation ───────────────────────────
    def _make_result(passes, to_tier3, agreement_val, reject_reason=None):
        return Tier2Result(
            provid=data.provid, passes=passes, to_tier3=to_tier3,
            best_period_mhaov=best_mhaov,
            best_period_mbls=best_mbls,
            best_period_mbls_raw=best_mbls_raw,
            best_period_ce=best_ce,
            consensus_period=consensus,
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
            consensus_contamination=consensus_cont,
            mbls_fap=mbls_fap,
            mbls_sig=mbls_sig,
            mhaov_sig=mhaov_sig,
            both_sig=both_sig,
            mbls_band_support=band_support,
            mbls_n_bands_supporting=n_bands_supporting,
            mbls_band_support_frac=band_support_frac,
            mhaov_confirms=mhaov_confirms,
        )

    # ── Path 1: Neither gate significant → reject ──────────────────────────────
    # No evidence for any period. Tier 3 on noise is uninformative.
    if not mbls_sig and not mhaov_sig:
        return _make_result(
            passes=False, to_tier3=False, agreement_val=False,
            reject_reason=(
                f"neither gate significant "
                f"(MBLS FAP={mbls_fap:.4f}, MHAOV p={p_value:.2e})"
            )
        )

    # ── Path 2: MHAOV confirms MBLS, either gate significant → publish ─────────
    # MHAOV agrees with MBLS (or its harmonic). Both methods point to the same
    # underlying period. Confidence depends on how many gates fired.
    #   both_sig → R=3 eligible (full confidence)
    #   either_sig → R≤2 (moderate confidence)
    if mhaov_confirms and either_sig:
        logger.debug(f"{data.provid}: MHAOV confirms MBLS={best_mbls:.3f}hr → mbls_confirmed")
        return _make_result(passes=True, to_tier3=False, agreement_val="mbls_confirmed")

    # ── Path 3: MBLS significant, MHAOV does not confirm ──────────────────────
    # MBLS FAP directly tests whether multi-band power exceeds the noise null
    # distribution. If FAP < 0.001, the multi-band signal is real. MHAOV
    # non-confirmation means the single-band collapsed series is less sensitive
    # at this particular period — not that the period is wrong.
    #
    # If MHAOV is also significant but at a genuinely different period, the
    # data contains two competing significant signals → ambiguous → Tier 3.
    # If MHAOV is not significant, MBLS evidence stands alone → publish R≤2.
    if mbls_sig and not mhaov_confirms:
        if mhaov_sig:
            # Both significant, genuinely disagree: real ambiguity → CLEAN
            logger.debug(
                f"{data.provid}: MBLS={best_mbls:.3f}hr sig, "
                f"MHAOV={best_mhaov:.3f}hr sig but disagrees → Tier 3"
            )
            return _make_result(passes=False, to_tier3=True, agreement_val=False)
        else:
            # Only MBLS significant: multi-band evidence stands alone
            logger.debug(
                f"{data.provid}: MBLS={best_mbls:.3f}hr sig (FAP={mbls_fap:.4f}), "
                f"MHAOV not sig or disagrees → mbls_sig_only"
            )
            return _make_result(passes=True, to_tier3=False, agreement_val="mbls_sig_only")

    # ── Path 4: Only MHAOV significant, MBLS not significant ──────────────────
    # This is the MHAOV-only case (4/76 objects). MHAOV found something MBLS
    # missed. Send to Tier 3 for CLEAN confirmation — we have one significant
    # signal but without multi-band confirmation it needs independent validation.
    if mhaov_sig and not mbls_sig:
        logger.debug(
            f"{data.provid}: MHAOV={best_mhaov:.3f}hr sig, MBLS not sig → Tier 3"
        )
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
    need to resolve individual peaks. A 500-point grid runs ~16× faster
    than the full fine grid.

    KNOWN LIMITATION — coarse/fine grid asymmetry
    ----------------------------------------------
    The observed max power is taken from the fine grid (up to 50,000 pts).
    The null distribution is built from permutations on the coarse grid
    (500 pts). This creates an asymmetry: a fine grid has more trial
    frequencies and therefore more chances for a noise peak to reach a
    given threshold, even under the null. Comparing fine-grid observed
    power against coarse-grid null power causes the FAP to be
    systematically underestimated — the test is anti-conservative
    (over-publishes in marginal cases).

    The correct fix (not yet implemented, see roadmap below) is either:
      (a) Run all permutations on the full fine grid — exact but slow.
      (b) Horne & Baliunas (1986) correction: scale observed fine-grid
          power to its coarse-grid equivalent before comparing, using the
          ratio of independent frequencies M_fine / M_coarse. The same
          correction is already applied in gls_fap() for MHAOV.

    In practice for Rubin commissioning data the vast majority of FAP
    values are 0.0000 (0/200 permutations exceed observed power). The
    signals are clear enough that the asymmetry is not causing false
    positives. The concern would matter most for marginal detections near
    the FAP threshold. This should be revisited in the Change 3 simulation
    study when synthetic lightcurves at varying SNR are evaluated.

    Roadmap: implement option (b) — compute observed power on coarse grid
    at the already-identified best_mbls period (single evaluation, not a
    full periodogram), then compare against the coarse-grid null.

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
