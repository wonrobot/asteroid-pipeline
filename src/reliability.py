"""
reliability.py
--------------
Computes an automated reliability code for each asteroid period result.

Analogous to the LCDB U code system (Warner et al.) but computed
automatically from the pipeline outputs, data characterisation, and
statistical diagnostics.

R codes
-------
R=3  : High confidence. All methods agree, signal significant,
       data regime supports full pipeline. Safe to publish.

R=2  : Moderate confidence. Methods agree but data is sparse or
       p-value is marginal. Publish with uncertainty caveat.

R=1  : Low confidence. Methods partially agree or Tier 3 tentative.
       Include in catalog but flag as uncertain.

R=0  : Insufficient confidence. Do not publish period.

R=-1 : Alias suspect. Result is close to a known alias frequency
       OR sits on a cadence-specific window peak.
       Requires independent confirmation before publishing.

Alias detection (two-layer)
---------------------------
Layer 1 — fixed list (daily/annual): flag_alias_risk()
  Checks the adopted period against known ground-based aliases:
  0.5 day, 1 day, 2 day, 0.5 year, 1 year.

Layer 2 — cadence-specific window contamination: flag_window_alias()
  Uses the spectral window function stored in t1result.window_power
  to compute a contamination score for the adopted period against the
  ACTUAL observation cadence of this specific asteroid. Catches
  dataset-specific aliases (e.g. 3-day LSST scheduling gaps, moon
  avoidance windows) that the fixed list misses entirely.

Both layers run independently. Either can trigger R=-1.

LCDB comparison
---------------
When a LCDB record exists (u_code >= 2), we additionally compute
lcdb_agreement and flag discrepancies.

Functions
---------
compute_reliability(char, t1, t2, t3, lcdb_record) — main entry point
flag_alias_risk(period_hr, tolerance)               — fixed alias check
flag_window_alias(period_hr, t1result)              — cadence alias check
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Known problematic alias periods in hours — fixed list
ALIAS_PERIODS_HR = [
    12.0,          # 0.5 day
    24.0,          # 1.0 day
    48.0,          # 2.0 days
    8760.0 / 2,    # 0.5 year
    8760.0,        # 1.0 year
]
ALIAS_TOLERANCE = 0.05   # 5%

# Window function contamination score used for the R-code cap (not R=-1 veto).
# This is NOT a detection threshold — it does not veto a period on its own.
# It is used only to cap R at 2 when contamination is high AND the significance
# gates are marginal. See _compute_r_code() and flag_window_alias() for usage.
#
# Value: 0.5 is the natural midpoint of the [0,1] contamination scale and
# corresponds to "the window power at this period is at least half the maximum
# window peak". This is a conservative cap trigger — it will only fire for
# periods that sit on genuinely prominent window peaks, not minor ripple.
# No empirical tuning has been applied to this value.
WINDOW_ALIAS_THRESHOLD = 0.5


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ReliabilityAssessment:
    """
    Full reliability assessment for one asteroid period result.

    Attributes
    ----------
    provid              : asteroid designation
    r_code              : reliability code (-1, 0, 1, 2, 3)
    r_flag              : string version e.g. "3", "2", "1", "0", "-1(alias)"
    period_hr           : adopted period in hours (NaN if r_code <= 0)
    period_unc_hr       : period uncertainty in hours
    source              : which tier/method provided the period
    agreement           : True if all Tier 2 methods agreed
    period_spread_pct   : max fractional spread across Tier 2 methods
    p_value             : MHAOV p-value at best period
    alias_risk          : True if fixed-list alias detected
    alias_note          : which fixed alias is nearby
    window_alias_risk   : True if cadence-specific alias detected
    window_alias_note   : contamination score and description
    lcdb_agreement      : "exact"/"half_period"/"double_period"/"disagree"/"no_prior"
    lcdb_delta_pct      : fractional difference from LCDB period
    notes               : human-readable explanation of the score
    """
    provid:             str
    r_code:             int
    r_flag:             str
    period_hr:          float
    period_unc_hr:      float
    source:             str
    agreement:          bool
    period_spread_pct:  float
    p_value:            float
    alias_risk:         bool
    alias_note:         str
    window_alias_risk:  bool     # NEW: cadence-specific alias
    window_alias_note:  str      # NEW: contamination score + description
    lcdb_agreement:     str
    lcdb_delta_pct:     float
    notes:              str


# ── Main function ─────────────────────────────────────────────────────────────

def compute_reliability(
    char,
    t1result,
    t2result=None,
    t3result=None,
    lcdb_record=None,
) -> ReliabilityAssessment:
    """
    Compute reliability code for one asteroid period result.
    """
    provid = char.provid

    # ── Case 1: Tier 1 failed ─────────────────────────────────────────────────
    if not t1result.passes:
        return ReliabilityAssessment(
            provid=provid, r_code=0, r_flag="0",
            period_hr=np.nan, period_unc_hr=np.nan,
            source="none", agreement=False,
            period_spread_pct=np.nan, p_value=np.nan,
            alias_risk=False, alias_note="",
            window_alias_risk=False, window_alias_note="",
            lcdb_agreement="no_prior", lcdb_delta_pct=np.nan,
            notes=f"Tier 1 failed: {t1result.reject_reason}",
        )

    # ── Case 2: Tier 2 not run ────────────────────────────────────────────────
    if t2result is None:
        return ReliabilityAssessment(
            provid=provid, r_code=0, r_flag="0",
            period_hr=np.nan, period_unc_hr=np.nan,
            source="none", agreement=False,
            period_spread_pct=np.nan, p_value=np.nan,
            alias_risk=False, alias_note="",
            window_alias_risk=False, window_alias_note="",
            lcdb_agreement="no_prior", lcdb_delta_pct=np.nan,
            notes="Tier 2 not run",
        )

    # ── Extract Tier 2 diagnostics ────────────────────────────────────────────
    agreement        = t2result.agreement
    # New agreement values: "mbls_confirmed" | "mbls_sig_only" | False
    # (old "True" and "two_of_three" no longer produced by tier2)
    period_spread    = t2result.period_spread_pct
    mbls_raw         = getattr(t2result, 'best_period_mbls_raw', t2result.best_period_mbls)
    p_value          = t2result.p_value
    consensus_period = t2result.consensus_period

    # ── Extract Tier 3 diagnostics ────────────────────────────────────────────
    t3_period     = np.nan
    t3_ci_width   = np.nan
    t3_peak_ratio = np.nan
    t3_reliable   = False

    if t3result is not None:
        t3_period     = t3result.best_period_adopted
        t3_ci_width   = t3result.ci_width
        t3_peak_ratio = t3result.clean_peak_ratio
        t3_reliable   = t3result.publish_tentative

    # ── Determine adopted period and source ───────────────────────────────────
    if t2result.passes:
        adopted_period = consensus_period
        period_unc     = np.nan
        source         = "tier2_consensus"
    elif t3result is not None and t3_reliable:
        adopted_period = t3_period
        period_unc     = t3_ci_width / 2.0
        source         = "tier3_bayesian"
    elif t3result is not None and not t3_reliable:
        adopted_period = np.nan
        period_unc     = np.nan
        source         = "none"
    else:
        adopted_period = np.nan
        period_unc     = np.nan
        source         = "none"

    # ── Layer 1: Fixed alias list ─────────────────────────────────────────────
    alias_risk, alias_note = flag_alias_risk(adopted_period)

    # ── Layer 2: Cadence-specific window alias ────────────────────────────────
    # Uses the actual window function computed from this asteroid's observation
    # timestamps — catches scheduling gaps and moon avoidance windows that the
    # fixed daily/annual list misses entirely.
    # baseline_hr from char.baseline_days × 24 — enables Rayleigh-criterion
    # radius in contamination_score (Change 3).
    char_baseline_hr = getattr(char, 'baseline_days', 0.0) * 24.0
    window_alias_risk, window_alias_note = flag_window_alias(
        adopted_period, t1result, baseline_hr=char_baseline_hr
    )

    # Combined alias flag
    # ─────────────────────────────────────────────────────────────────────────
    # Layer 1 (fixed list) can independently trigger R=-1. The physical causes
    # are universal and well-understood: Earth rotation (24hr, 12hr harmonics),
    # Earth orbital period (365.25d, 182.6d). Citation: VanderPlas (2018 PASP).
    #
    # Layer 2 (window function) does NOT independently trigger R=-1.
    #
    # Scientific basis: a high contamination score means the window function
    # peaks near the adopted period, but "the window peaks here" and "the signal
    # is an alias" are not the same statement. The signal could be real at a
    # period that happens to coincide with a cadence gap — this is expected for
    # any sufficiently dense period grid. The MBLS FAP already tests whether the
    # observed power exceeds the noise null distribution. If the FAP is low, that
    # is direct statistical evidence for a real signal, which outweighs the
    # indirect evidence from window contamination.
    #
    # Combining both as an OR-veto would implicitly weight them equally, which
    # requires a calibrated ROC analysis on simulated data to justify — exactly
    # the kind of post-hoc threshold tuning the project avoids. See Change 3 in
    # the README (planned: derive thresholds from recovery rate curves).
    #
    # Layer 2 is instead used in two weaker roles:
    #   (a) r_flag annotation ("cadence_alias") — visible in the catalog
    #   (b) R-code cap inside _compute_r_code() when contamination is high AND
    #       the significance evidence is not strong (caps R at 2, not at -1)
    #
    # The one case where Layer 2 DOES join the hard veto: when the window shows
    # high contamination AND neither significance gate fires. In that case there
    # is no independent evidence for the period at all — both the window and the
    # FAP agree there is no signal, so R=-1 is warranted.
    mhaov_sig  = getattr(t2result, 'mhaov_sig', False) if t2result is not None else False
    mbls_sig   = getattr(t2result, 'mbls_sig',  False) if t2result is not None else False
    either_sig = mhaov_sig or mbls_sig

    # Hard alias flag: Layer 1 always counts; Layer 2 only when no sig support
    any_alias = alias_risk or (window_alias_risk and not either_sig)

    # ── LCDB comparison ───────────────────────────────────────────────────────
    lcdb_agreement  = "no_prior"
    lcdb_delta_pct  = np.nan

    if (lcdb_record is not None
            and lcdb_record.found
            and not np.isnan(lcdb_record.period_hr)
            and not np.isnan(adopted_period)):
        from sources.lcdb import compare_to_lcdb
        cmp            = compare_to_lcdb(adopted_period, lcdb_record)
        lcdb_agreement = cmp["agreement"]
        lcdb_delta_pct = cmp["delta_pct"] * 100

    # ── Compute R code ────────────────────────────────────────────────────────
    r_code, notes = _compute_r_code(
        regime              = char.regime,
        reliability_ceiling = char.reliability_ceiling,
        agreement           = agreement,
        period_spread       = period_spread,
        p_value             = p_value,
        t2_passes           = t2result.passes,
        t3_reliable         = t3_reliable,
        t3_ci_width         = t3_ci_width,
        t3_peak_ratio       = t3_peak_ratio,
        alias_risk          = any_alias,
        alias_note          = alias_note or window_alias_note,
        adopted_period      = adopted_period,
        n_obs               = char.n_obs,
        n_nights            = char.n_nights,
        mbls_raw            = mbls_raw,
        consensus_contamination = getattr(t2result, 'consensus_contamination', np.nan),
        both_sig            = getattr(t2result, 'both_sig', False),
        mbls_fap            = getattr(t2result, 'mbls_fap', np.nan),
        mbls_band_support_frac    = getattr(t2result, 'mbls_band_support_frac', 0.0),
        mbls_n_bands_supporting   = getattr(t2result, 'mbls_n_bands_supporting', 0),
    )

    # ── Build r_flag string ───────────────────────────────────────────────────
    # Layer 1 (fixed list) alias → always annotated, triggers R=-1
    # Layer 2 (window) + no sig → hard veto, annotated as cadence_alias
    # Layer 2 (window) + sig present → soft note only, R capped at 2 internally
    r_flag = str(r_code)
    if alias_risk:
        # Fixed-list alias: always a hard flag
        r_flag = f"{r_code}-alias"
    elif window_alias_risk and not either_sig:
        # Window alias with no significance: hard veto (any_alias=True → r_code=-1)
        r_flag = f"{r_code}-cadence_alias"
    elif window_alias_risk and either_sig:
        # Window alias but signal IS significant: soft annotation only
        # r_code already capped at 2 by _compute_r_code(); flag it for catalog
        r_flag = f"{r_code}-cadence_alias_soft"

    logger.debug(
        f"{provid}: R={r_code} ({r_flag}) P={adopted_period:.3f}hr "
        f"agree={agreement} p={p_value:.2e} "
        f"fixed_alias={alias_risk} window_alias={window_alias_risk}"
    )

    return ReliabilityAssessment(
        provid            = provid,
        r_code            = r_code,
        r_flag            = r_flag,
        period_hr         = adopted_period,
        period_unc_hr     = period_unc,
        source            = source,
        agreement         = agreement,
        period_spread_pct = period_spread,
        p_value           = p_value,
        alias_risk        = alias_risk,
        alias_note        = alias_note,
        window_alias_risk = window_alias_risk,
        window_alias_note = window_alias_note,
        lcdb_agreement    = lcdb_agreement,
        lcdb_delta_pct    = lcdb_delta_pct,
        notes             = notes,
    )


# ── R code logic ──────────────────────────────────────────────────────────────

def _compute_r_code(
    regime, reliability_ceiling, agreement, period_spread,
    p_value, t2_passes, t3_reliable, t3_ci_width,
    t3_peak_ratio, alias_risk, alias_note, adopted_period,
    n_obs, n_nights, mbls_raw=None,
    consensus_contamination=np.nan,
    both_sig=False,
    mbls_fap=np.nan,
    mbls_band_support_frac=0.0,
    mbls_n_bands_supporting=0,
) -> tuple:
    """
    Core R code decision logic. Returns (r_code, notes_string).

    Agreement values (new):
      "mbls_confirmed"  — MHAOV confirmed MBLS period + either gate significant
      "mbls_sig_only"   — MBLS significant, MHAOV did not confirm (not sig or disagrees)
      False             — rejected or routed to Tier 3

    Contamination modifier
    ----------------------
    When the consensus period has high window contamination
    (consensus_contamination > WINDOW_ALIAS_THRESHOLD) but is not
    already flagged as an alias, we cap R at 2. The period may still be
    real — the contamination means we can't rule out the cadence alias.
    """
    ceiling_map = {"high": 3, "medium": 2, "low": 1, "unknown": 1}
    ceiling     = ceiling_map.get(reliability_ceiling, 1)

    SPIN_BARRIER_HR = 2.2
    superfast_raw = (
        mbls_raw is not None
        and not np.isnan(mbls_raw)
        and mbls_raw < SPIN_BARRIER_HR
        and regime not in ('rich_multiyear', 'combined')
    )
    superfast_adopted = (
        not np.isnan(adopted_period)
        and adopted_period < SPIN_BARRIER_HR
        and regime not in ('rich_multiyear', 'combined')
    )
    if superfast_raw or superfast_adopted:
        ceiling = min(ceiling, 2)

    # Contamination cap
    if (not np.isnan(consensus_contamination)
            and consensus_contamination > WINDOW_ALIAS_THRESHOLD
            and not alias_risk):
        ceiling = min(ceiling, 2)

    # No period
    if np.isnan(adopted_period):
        if not t2_passes and not t3_reliable:
            return 0, _note(
                "No reliable period. Tier 2 did not pass and "
                "Tier 3 could not resolve ambiguity."
            )
        return 0, _note("No period adopted.")

    # Alias risk overrides everything
    if alias_risk:
        return -1, _note(
            f"Period {adopted_period:.3f}hr is close to known alias: "
            f"{alias_note}. Independent confirmation required."
        )

    # ── T2 published ──────────────────────────────────────────────────────────
    if t2_passes:

        # ── Path A: MHAOV confirmed MBLS ──────────────────────────────────────
        # Both methods point to the same period. R=3 requires both gates and
        # a dense/rich regime. R=2 otherwise.
        if agreement == "mbls_confirmed":
            if regime in ("dense", "rich_multiyear", "combined"):
                if both_sig:
                    r = 3
                    note = _note(
                        f"MHAOV confirms MBLS period (spread={period_spread*100:.1f}%), "
                        f"both gates significant "
                        f"(MHAOV p={p_value:.2e}, MBLS FAP={mbls_fap:.4f}). "
                        f"Dense regime. High confidence."
                    )
                elif p_value < 0.001:
                    r = 2
                    note = _note(
                        f"MHAOV confirms MBLS period (spread={period_spread*100:.1f}%), "
                        f"MHAOV p={p_value:.2e} significant, "
                        f"MBLS FAP={mbls_fap:.4f} marginal. Moderate confidence."
                    )
                elif not np.isnan(mbls_fap) and mbls_fap < 0.001:
                    r = 2
                    note = _note(
                        f"MHAOV confirms MBLS period (spread={period_spread*100:.1f}%), "
                        f"MBLS FAP={mbls_fap:.4f} significant (multi-band), "
                        f"MHAOV p={p_value:.2e} marginal. Moderate confidence."
                    )
                else:
                    r = 1
                    note = _note(
                        f"MHAOV confirms MBLS period but neither gate strongly significant "
                        f"(MHAOV p={p_value:.2e}, MBLS FAP={mbls_fap:.4f}). Tentative."
                    )
            else:
                # sparse / unknown regime
                if both_sig:
                    r = 2
                    note = _note(
                        f"MHAOV confirms MBLS period (spread={period_spread*100:.1f}%), "
                        f"both gates significant (MHAOV p={p_value:.2e}, "
                        f"MBLS FAP={mbls_fap:.4f}), sparse regime. Moderate confidence."
                    )
                elif p_value < 0.001 or (not np.isnan(mbls_fap) and mbls_fap < 0.001):
                    r = 2
                    note = _note(
                        f"MHAOV confirms MBLS period (spread={period_spread*100:.1f}%), "
                        f"one gate significant (MHAOV p={p_value:.2e}, "
                        f"MBLS FAP={mbls_fap:.4f}), sparse regime. Moderate confidence."
                    )
                else:
                    r = 1
                    note = _note(
                        f"MHAOV confirms MBLS period but sparse data and both gates marginal "
                        f"(MHAOV p={p_value:.2e}, MBLS FAP={mbls_fap:.4f}). Low confidence."
                    )

        # ── Path B: MBLS significant, MHAOV does not confirm ─────────────────
        # MBLS FAP directly tests whether multi-band power exceeds the noise
        # null. FAP < 0.001 is real evidence for a period. MHAOV non-confirmation
        # means the single-band collapsed series is less sensitive here — not that
        # the period is wrong. R capped at 2 (no independent confirmation).
        elif agreement == "mbls_sig_only":
            if not np.isnan(mbls_fap) and mbls_fap < 0.001:
                r = 2
                note = _note(
                    f"MBLS significant (FAP={mbls_fap:.4f}) — multi-band evidence. "
                    f"MHAOV (p={p_value:.2e}) does not confirm this period; "
                    f"single-band series less sensitive. "
                    f"Period adopted from MBLS. Moderate confidence (R=2, no independent confirmation)."
                )
            else:
                r = 1
                note = _note(
                    f"MBLS marginally significant (FAP={mbls_fap:.4f}), "
                    f"MHAOV (p={p_value:.2e}) does not confirm. Low confidence."
                )

        else:
            # Fallback — should not normally reach here
            r = 1
            note = _note(f"T2 passed with unrecognised agreement value '{agreement}'.")

        # Append contamination note if ceiling was capped
        if (not np.isnan(consensus_contamination)
                and consensus_contamination > WINDOW_ALIAS_THRESHOLD):
            note += (f" [R capped at 2: consensus period has window "
                     f"contamination={consensus_contamination:.2f}]")

        return min(r, ceiling), note

    # T3 result
    if t3_reliable:
        ci_narrow   = not np.isnan(t3_ci_width) and t3_ci_width < 0.5
        clean_good  = not np.isnan(t3_peak_ratio) and t3_peak_ratio >= 3.0

        if ci_narrow and clean_good:
            r    = 2
            note = _note(
                f"Tier 3 tentative: CI={t3_ci_width:.3f}hr, "
                f"CLEAN ratio={t3_peak_ratio:.1f}. Both criteria met."
            )
        else:
            r    = 1
            note = _note(
                f"Tier 3 tentative: CI={t3_ci_width:.3f}hr, "
                f"CLEAN ratio={t3_peak_ratio:.1f}. Only one criterion met."
            )
        return min(r, ceiling), note

    return 0, _note(
        "Tier 2 methods disagree and Tier 3 could not resolve. "
        "Period not published. Flag for follow-up."
    )


def _note(text: str) -> str:
    return text.strip()


# ── Layer 1: Fixed alias check ────────────────────────────────────────────────

def flag_alias_risk(
    period_hr: float,
    tolerance: float = ALIAS_TOLERANCE,
) -> tuple:
    """
    Check if a period is close to a known fixed alias frequency.
    Returns (is_risky, note_string).
    """
    if np.isnan(period_hr) or period_hr <= 0:
        return False, ""

    alias_names = {
        12.0:          "0.5-day alias",
        24.0:          "1-day alias",
        48.0:          "2-day alias",
        8760.0 / 2:    "0.5-year alias",
        8760.0:        "1-year alias",
    }

    for alias_p, name in alias_names.items():
        delta = abs(period_hr - alias_p) / alias_p
        if delta <= tolerance:
            return True, f"{name} ({alias_p:.1f}hr, Δ={delta*100:.1f}%)"

    return False, ""


# ── Layer 2: Cadence-specific window alias check ──────────────────────────────

def flag_window_alias(
    period_hr:   float,
    t1result,
    baseline_hr: float = 0.0,
    threshold:   float = WINDOW_ALIAS_THRESHOLD,
) -> tuple:
    """
    Check if a period sits on a peak in the spectral window function —
    meaning the sampling cadence of THIS specific dataset can produce
    that period as an alias, regardless of whether any real signal exists.

    This is more informative than the fixed list because:
    - Every asteroid has a unique observing history (different nights, gaps)
    - LSST scheduling introduces dataset-specific gaps (moon, weather,
      field rotation) that appear at different periods for each object
    - The window function directly encodes these gaps as alias peaks

    Parameters
    ----------
    period_hr : adopted period to check
    t1result  : Tier1Result containing window_power and test_periods
    threshold : contamination score above which we flag as alias risk

    Returns
    -------
    (is_risky, note_string)
    """
    if np.isnan(period_hr) or period_hr <= 0:
        return False, ""

    # Guard: window_power may be empty for rejected objects
    if (not hasattr(t1result, 'window_power')
            or len(t1result.window_power) == 0
            or len(t1result.test_periods) == 0):
        return False, ""

    from window import contamination_score
    score = contamination_score(
        period_hr,
        t1result.test_periods,
        t1result.window_power,
        baseline_hr=baseline_hr,
    )

    if score > threshold:
        return True, (
            f"cadence alias: period {period_hr:.3f}hr sits on window peak "
            f"(contamination={score:.2f} > {threshold}). "
            f"This alias arises from the specific observation gaps for this "
            f"asteroid — not in fixed daily/annual list."
        )

    return False, ""
