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
       Equivalent to LCDB U=3.

R=2  : Moderate confidence. Methods agree but data is sparse or
       p-value is marginal. Publish with uncertainty caveat.
       Equivalent to LCDB U=2.

R=1  : Low confidence. Methods partially agree or Tier 3 tentative.
       Include in catalog but flag as uncertain.
       Equivalent to LCDB U=1.

R=0  : Insufficient confidence. Methods disagree and Tier 3 cannot
       resolve. Do not publish period. Flag for follow-up.

R=-1 : Alias suspect. Result is close to a known alias frequency
       (0.5 day, 1 day, 0.5 year, 1 year) regardless of agreement.
       Requires independent confirmation before publishing.

LCDB comparison
---------------
When a LCDB record exists (u_code >= 2), we additionally compute
lcdb_agreement and flag discrepancies as scientifically interesting:

  exact        : pipeline agrees within 5% — confirms LCDB
  half_period  : pipeline found P/2 — double-hump alias
  double_period: pipeline found 2P  — possible LCDB error
  disagree     : significant disagreement — investigate

Functions
---------
compute_reliability(char, t1, t2, t3, lcdb_record) — main entry point
flag_alias_risk(period_hr, tolerance)               — alias frequency check
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Known problematic alias periods in hours
ALIAS_PERIODS_HR = [
    12.0,          # 0.5 day
    24.0,          # 1.0 day
    48.0,          # 2.0 days
    8760.0 / 2,    # 0.5 year
    8760.0,        # 1.0 year
]
ALIAS_TOLERANCE = 0.05   # 5% — period within this of alias → flag


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ReliabilityAssessment:
    """
    Full reliability assessment for one asteroid period result.

    Attributes
    ----------
    provid           : asteroid designation
    r_code           : reliability code (-1, 0, 1, 2, 3)
    r_flag           : string version e.g. "3", "2", "1", "0", "-1(alias)"
    period_hr        : adopted period in hours (NaN if r_code <= 0)
    period_unc_hr    : period uncertainty in hours (NaN if not available)
    source           : which tier/method provided the period
    agreement        : True if all Tier 2 methods agreed
    period_spread_pct: max fractional spread across Tier 2 methods
    p_value          : MHAOV p-value at best period
    alias_risk       : True if period is close to a known alias
    alias_note       : which alias is nearby (empty if no risk)
    lcdb_agreement   : "exact"/"half_period"/"double_period"/"disagree"/"no_prior"
    lcdb_delta_pct   : fractional difference from LCDB period
    notes            : human-readable explanation of the score
    """
    provid:            str
    r_code:            int
    r_flag:            str
    period_hr:         float
    period_unc_hr:     float
    source:            str
    agreement:         bool
    period_spread_pct: float
    p_value:           float
    alias_risk:        bool
    alias_note:        str
    lcdb_agreement:    str
    lcdb_delta_pct:    float
    notes:             str


# ── Main function ─────────────────────────────────────────────────────────────

def compute_reliability(
    char,          # DataCharacterisation
    t1result,      # Tier1Result
    t2result=None, # Tier2Result | None
    t3result=None, # Tier3Result | None
    lcdb_record=None,  # LCDBRecord | None
) -> ReliabilityAssessment:
    """
    Compute reliability code for one asteroid period result.

    Parameters
    ----------
    char        : DataCharacterisation from characterise.py
    t1result    : Tier1Result
    t2result    : Tier2Result (None if Tier 1 failed)
    t3result    : Tier3Result (None if Tier 2 published or pipeline stopped)
    lcdb_record : LCDBRecord from sources.lcdb (optional)

    Returns
    -------
    ReliabilityAssessment
    """
    provid = char.provid

    # ── Case 1: Tier 1 failed — no period ─────────────────────────────────────
    if not t1result.passes:
        return ReliabilityAssessment(
            provid=provid, r_code=0, r_flag="0",
            period_hr=np.nan, period_unc_hr=np.nan,
            source="none", agreement=False,
            period_spread_pct=np.nan, p_value=np.nan,
            alias_risk=False, alias_note="",
            lcdb_agreement="no_prior", lcdb_delta_pct=np.nan,
            notes=f"Tier 1 failed: {t1result.reject_reason}",
        )

    # ── Case 2: Tier 2 not run ─────────────────────────────────────────────────
    if t2result is None:
        return ReliabilityAssessment(
            provid=provid, r_code=0, r_flag="0",
            period_hr=np.nan, period_unc_hr=np.nan,
            source="none", agreement=False,
            period_spread_pct=np.nan, p_value=np.nan,
            alias_risk=False, alias_note="",
            lcdb_agreement="no_prior", lcdb_delta_pct=np.nan,
            notes="Tier 2 not run",
        )

    # ── Extract Tier 2 diagnostics ────────────────────────────────────────────
    agreement        = t2result.agreement
    period_spread    = t2result.period_spread_pct
    p_value          = t2result.p_value
    consensus_period = t2result.consensus_period

    # ── Extract Tier 3 diagnostics if available ───────────────────────────────
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
        period_unc     = np.nan   # CI not computed at Tier 2
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

    # ── Alias risk check ──────────────────────────────────────────────────────
    alias_risk, alias_note = flag_alias_risk(adopted_period)

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
        regime           = char.regime,
        reliability_ceiling = char.reliability_ceiling,
        agreement        = agreement,
        period_spread    = period_spread,
        p_value          = p_value,
        t2_passes        = t2result.passes,
        t3_reliable      = t3_reliable,
        t3_ci_width      = t3_ci_width,
        t3_peak_ratio    = t3_peak_ratio,
        alias_risk       = alias_risk,
        alias_note       = alias_note,
        adopted_period   = adopted_period,
        n_obs            = char.n_obs,
        n_nights         = char.n_nights,
    )

    # ── Build r_flag string ───────────────────────────────────────────────────
    r_flag = str(r_code)
    if alias_risk and r_code > 0:
        r_flag = f"{r_code}-alias"

    logger.debug(
        f"{provid}: R={r_code} ({r_flag}) P={adopted_period:.3f}hr "
        f"agree={agreement} p={p_value:.2e} alias={alias_risk}"
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
        lcdb_agreement    = lcdb_agreement,
        lcdb_delta_pct    = lcdb_delta_pct,
        notes             = notes,
    )


# ── R code logic ──────────────────────────────────────────────────────────────

def _compute_r_code(
    regime, reliability_ceiling, agreement, period_spread,
    p_value, t2_passes, t3_reliable, t3_ci_width,
    t3_peak_ratio, alias_risk, alias_note, adopted_period,
    n_obs, n_nights,
) -> tuple:
    """
    Core R code decision logic. Returns (r_code, notes_string).

    Decision tree:
    ─────────────
    No period adopted → R=0
    Alias risk        → R=-1

    T2 passes:
      dense + agree + p<0.001   → R=3
      sparse + agree + p<0.001  → R=2
      any + agree + p<0.01      → R=2
      any + agree + p>=0.01     → R=1

    T3 tentative:
      ci_width < 0.5 AND peak_ratio > 3   → R=2
      ci_width < 0.5 OR  peak_ratio > 3   → R=1
      neither                              → R=0

    Ceiling cap: never exceed reliability_ceiling
      high   → max R=3
      medium → max R=2
      low    → max R=1
    """
    ceiling_map = {"high": 3, "medium": 2, "low": 1, "unknown": 1}
    ceiling     = ceiling_map.get(reliability_ceiling, 1)

    # No period
    if np.isnan(adopted_period):
        if not t2_passes and not t3_reliable:
            return 0, _note(
                "No reliable period. Methods disagree at Tier 2 and "
                "Tier 3 could not resolve ambiguity. "
                "More observations needed."
            )
        return 0, _note("No period adopted.")

    # Alias risk overrides everything
    if alias_risk:
        return -1, _note(
            f"Period {adopted_period:.3f}hr is close to known alias: "
            f"{alias_note}. Independent confirmation required."
        )

    # T2 published
    if t2_passes:
        if regime in ("dense", "rich_multiyear", "combined"):
            if p_value < 0.001:
                r = 3
                note = _note(
                    f"All 3 methods agree (spread={period_spread*100:.1f}%), "
                    f"p={p_value:.2e}, dense regime. High confidence."
                )
            elif p_value < 0.01:
                r = 2
                note = _note(
                    f"All 3 methods agree (spread={period_spread*100:.1f}%), "
                    f"p={p_value:.2e} marginal. Moderate confidence."
                )
            else:
                r = 1
                note = _note(
                    f"Methods agree but p={p_value:.2e} not significant. "
                    f"Treat as tentative."
                )
        else:  # sparse
            if p_value < 0.001:
                r = 2
                note = _note(
                    f"All 3 methods agree (spread={period_spread*100:.1f}%), "
                    f"p={p_value:.2e}, sparse regime. Moderate confidence."
                )
            else:
                r = 1
                note = _note(
                    f"Methods agree but sparse data and p={p_value:.2e}. "
                    f"Low confidence."
                )
        return min(r, ceiling), note

    # T3 result
    if t3_reliable:
        ci_narrow   = not np.isnan(t3_ci_width) and t3_ci_width < 0.5
        clean_good  = not np.isnan(t3_peak_ratio) and t3_peak_ratio >= 3.0

        if ci_narrow and clean_good:
            r    = 2
            note = _note(
                f"Tier 3 tentative: CI={t3_ci_width:.3f}hr, "
                f"CLEAN ratio={t3_peak_ratio:.1f}. "
                f"Both criteria met. Moderate confidence."
            )
        else:
            r    = 1
            note = _note(
                f"Tier 3 tentative: CI={t3_ci_width:.3f}hr, "
                f"CLEAN ratio={t3_peak_ratio:.1f}. "
                f"Only one criterion met. Low confidence."
            )
        return min(r, ceiling), note

    return 0, _note(
        "Tier 2 methods disagree and Tier 3 could not resolve. "
        "Period not published. Flag for follow-up."
    )


def _note(text: str) -> str:
    return text.strip()


# ── Alias risk ────────────────────────────────────────────────────────────────

def flag_alias_risk(
    period_hr: float,
    tolerance: float = ALIAS_TOLERANCE,
) -> tuple:
    """
    Check if a period is suspiciously close to a known alias frequency.

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
