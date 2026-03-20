"""
characterise.py
---------------
Characterises the data available for a single asteroid and classifies
it into a regime that determines which period-finding methods to trust.

Regimes
-------
dense
    Short baseline (<30 days), many obs per night (>20), single season.
    Typical of commissioning or targeted campaigns.
    All methods reliable. CE works well. CLEAN useful but not critical.
    Example: your current x05 commissioning data.

sparse
    Single season, few obs per night or low total count.
    CE unreliable (phase curve under-sampled).
    MBLS primary method. MHAOV for significance only.
    CLEAN important for alias identification.

rich_multiyear
    3+ seasons, 500+ total observations.
    Full pipeline reliable. CLEAN very powerful (window function known).
    Period uncertainty improves with baseline.

combined
    Data from multiple survey sources (LSST + ZTF etc).
    Different window functions suppress cross-survey aliases.
    CLEAN most powerful. All methods applicable.

unknown
    Insufficient data to classify. Flag for manual review.

Alias risk
----------
We flag periods close to known aliases from the LSST cadence:
    0.5 day, 1.0 day (daily aliases)
    0.5 yr, 1.0 yr  (annual aliases)
    Synodic month    (Moon avoidance gaps)

Functions
---------
characterise(df_obj, lcdb_record)   — main entry point
_count_seasons(mjds)                — count distinct observing seasons
_compute_alias_risk(period_hr)      — flag known alias frequencies
_classify_regime(...)               — regime classification logic
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# Known alias periods in hours from LSST/ground-based cadence
KNOWN_ALIASES_HR = [
    12.0,        # 0.5 day
    24.0,        # 1.0 day
    48.0,        # 2.0 day
    8760.0 / 2,  # 0.5 year
    8760.0,      # 1.0 year
]
ALIAS_TOL = 0.05   # 5% tolerance for alias flagging


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class DataCharacterisation:
    """
    Full characterisation of one asteroid's available data.

    Attributes
    ----------
    provid              : asteroid designation
    regime              : data regime classification
    n_obs               : total observations
    n_bands             : number of distinct bands
    bands               : list of band names
    baseline_days       : total time span in days
    n_nights            : number of distinct observing nights
    n_seasons           : number of distinct observing seasons
    obs_per_night_median: median observations per night
    obs_per_night_max   : maximum observations per night
    night_duration_hr   : median duration of each observing night in hours
    snr_proxy           : amplitude / median(rmsmag) proxy
    mag_range           : peak-to-peak raw magnitude range
    dominant_aliases_hr : list of alias periods (hours) to watch for
    alias_risk          : dict of {period_hr: risk_label} for known aliases
    lcdb_period         : known period from LCDB (NaN if not found)
    lcdb_u_code         : LCDB reliability code (0 if not found)
    recommended_methods : methods appropriate for this regime
    reliability_ceiling : best achievable reliability given data quality
    notes               : human-readable summary
    """
    provid:               str
    regime:               str
    n_obs:                int
    n_bands:              int
    bands:                List[str]
    baseline_days:        float
    n_nights:             int
    n_seasons:            int
    obs_per_night_median: float
    obs_per_night_max:    int
    night_duration_hr:    float
    snr_proxy:            float
    mag_range:            float
    dominant_aliases_hr:  List[float]
    alias_risk:           dict
    lcdb_period:          float
    lcdb_u_code:          int
    recommended_methods:  List[str]
    reliability_ceiling:  str
    notes:                str


# ── Regime → method mapping ───────────────────────────────────────────────────

REGIME_METHODS = {
    "dense":          ["gls", "mbls", "mhaov", "ce", "clean"],
    "sparse":         ["mbls", "mhaov", "clean"],
    "rich_multiyear": ["mbls", "mhaov", "ce", "clean"],
    "combined":       ["mbls", "mhaov", "ce", "clean"],
    "unknown":        ["mbls"],
}

REGIME_CEILING = {
    "dense":          "high",     # all methods, good phase coverage
    "sparse":         "medium",   # MBLS+MHAOV only, alias risk elevated
    "rich_multiyear": "high",     # full pipeline, long baseline
    "combined":       "high",     # best alias breaking
    "unknown":        "low",      # flag for review
}


# ── Main function ─────────────────────────────────────────────────────────────

def characterise(
    df_obj:       pd.DataFrame,
    lcdb_record   = None,       # LCDBRecord | None
) -> DataCharacterisation:
    """
    Characterise the data available for a single asteroid.

    Parameters
    ----------
    df_obj      : observations DataFrame (from ingestion)
                  Required columns: provid, mjd, band, mag, rmsmag
    lcdb_record : LCDBRecord from sources.lcdb (optional)

    Returns
    -------
    DataCharacterisation
    """
    provid = df_obj["provid"].iloc[0] if "provid" in df_obj.columns else "unknown"

    # Normalize band names
    BAND_REMAP = {"g":"Lg","r":"Lr","i":"Li","z":"Lz","y":"Ly","u":"Lu"}
    df = df_obj.copy()
    df["band"] = df["band"].replace(BAND_REMAP)
    df = df.sort_values("mjd").reset_index(drop=True)

    mjds  = df["mjd"].values
    mags  = df["mag"].values
    errs  = df["rmsmag"].values

    # ── Basic counts ─────────────────────────────────────────────────────────
    n_obs        = len(df)
    bands        = sorted(df["band"].unique().tolist())
    n_bands      = len(bands)
    baseline_days = float(mjds.max() - mjds.min()) if n_obs > 1 else 0.0

    # ── Night structure ───────────────────────────────────────────────────────
    night_ids = _assign_nights(mjds)
    df["_night"] = night_ids
    n_nights = int(night_ids.max()) + 1

    night_stats = df.groupby("_night").agg(
        n_obs     = ("mjd", "count"),
        duration  = ("mjd", lambda x: (x.max() - x.min()) * 24.0),
    )
    obs_per_night_median = float(night_stats["n_obs"].median())
    obs_per_night_max    = int(night_stats["n_obs"].max())
    night_duration_hr    = float(night_stats["duration"].median())

    # ── Season structure ──────────────────────────────────────────────────────
    n_seasons = _count_seasons(mjds)

    # ── SNR proxy ─────────────────────────────────────────────────────────────
    mag_range   = float(mags.max() - mags.min()) if n_obs > 1 else 0.0
    median_err  = float(np.median(errs)) if n_obs > 0 else 1.0
    snr_proxy   = mag_range / median_err if median_err > 0 else 0.0

    # ── Regime classification ─────────────────────────────────────────────────
    # TODO: n_sources hardcoded to 1 — "combined" regime is currently unreachable.
    # When adding ZTF or other surveys, pass n_sources here and update ingestion.py
    # to tag observations by source before calling characterise().
    n_sources = 1
    regime = _classify_regime(
        n_obs=n_obs,
        baseline_days=baseline_days,
        n_nights=n_nights,
        n_seasons=n_seasons,
        obs_per_night_median=obs_per_night_median,
        n_sources=n_sources,
    )

    # ── Alias risk ────────────────────────────────────────────────────────────
    dominant_aliases_hr = KNOWN_ALIASES_HR.copy()
    alias_risk = _compute_alias_risk(dominant_aliases_hr)

    # ── LCDB prior ────────────────────────────────────────────────────────────
    lcdb_period = np.nan
    lcdb_u_code = 0
    if lcdb_record is not None and lcdb_record.found:
        lcdb_period = lcdb_record.period_hr
        lcdb_u_code = lcdb_record.u_code

    # ── Methods and ceiling ───────────────────────────────────────────────────
    recommended_methods = REGIME_METHODS[regime]
    reliability_ceiling = REGIME_CEILING[regime]

    # Downgrade ceiling if data quality is marginal
    if n_obs < 30:
        reliability_ceiling = "low"
        recommended_methods = ["mbls"]
    elif n_obs < 60 and regime == "dense":
        reliability_ceiling = "medium"

    # ── Notes ─────────────────────────────────────────────────────────────────
    notes = _build_notes(
        regime=regime,
        n_obs=n_obs,
        n_nights=n_nights,
        n_seasons=n_seasons,
        baseline_days=baseline_days,
        obs_per_night_median=obs_per_night_median,
        night_duration_hr=night_duration_hr,
        snr_proxy=snr_proxy,
        lcdb_period=lcdb_period,
        lcdb_u_code=lcdb_u_code,
        reliability_ceiling=reliability_ceiling,
    )

    logger.debug(f"{provid}: regime={regime} ceiling={reliability_ceiling} "
                 f"N={n_obs} nights={n_nights} seasons={n_seasons}")

    return DataCharacterisation(
        provid               = provid,
        regime               = regime,
        n_obs                = n_obs,
        n_bands              = n_bands,
        bands                = bands,
        baseline_days        = baseline_days,
        n_nights             = n_nights,
        n_seasons            = n_seasons,
        obs_per_night_median = obs_per_night_median,
        obs_per_night_max    = obs_per_night_max,
        night_duration_hr    = night_duration_hr,
        snr_proxy            = snr_proxy,
        mag_range            = mag_range,
        dominant_aliases_hr  = dominant_aliases_hr,
        alias_risk           = alias_risk,
        lcdb_period          = lcdb_period,
        lcdb_u_code          = lcdb_u_code,
        recommended_methods  = recommended_methods,
        reliability_ceiling  = reliability_ceiling,
        notes                = notes,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _assign_nights(mjds: np.ndarray, gap_days: float = 0.3) -> np.ndarray:
    """
    Assign integer night IDs to observations.
    A new night starts when the gap between consecutive observations
    exceeds gap_days (default 0.3 days = 7.2 hours).
    """
    night_ids  = np.zeros(len(mjds), dtype=int)
    night      = 0
    prev_mjd   = mjds[0]
    for i, mjd in enumerate(mjds):
        if mjd - prev_mjd > gap_days:
            night += 1
        night_ids[i] = night
        prev_mjd = mjd
    return night_ids


def _count_seasons(mjds: np.ndarray, season_gap_days: float = 60.0) -> int:
    """
    Count distinct observing seasons.
    A new season starts when the gap between observations exceeds
    season_gap_days (default 60 days).
    """
    if len(mjds) == 0:
        return 0
    sorted_mjds = np.sort(mjds)
    gaps        = np.diff(sorted_mjds)
    n_seasons   = int(np.sum(gaps > season_gap_days)) + 1
    return n_seasons


def _classify_regime(
    n_obs:                int,
    baseline_days:        float,
    n_nights:             int,
    n_seasons:            int,
    obs_per_night_median: float,
    n_sources:            int,
) -> str:
    """
    Classify data into a regime string.

    Decision tree:
    1. Multiple sources → combined (best alias breaking)
    2. Multiple seasons (>=3) and rich data → rich_multiyear
    3. Short baseline, dense nightly coverage → dense
    4. Otherwise → sparse
    5. Too few obs → unknown
    """
    if n_obs < 20:
        return "unknown"

    if n_sources > 1:
        return "combined"

    if n_seasons >= 3 and n_obs >= 500:
        return "rich_multiyear"

    # Dense: short baseline, good nightly coverage, multiple nights
    if (baseline_days <= 30
            and n_nights >= 3
            and obs_per_night_median >= 20
            and n_obs >= 60):
        return "dense"

    return "sparse"


def _compute_alias_risk(alias_periods_hr: list) -> dict:
    """
    For each known alias period, return a risk label.
    Used to warn when a pipeline result is suspiciously close to an alias.
    """
    risk = {}
    for p in alias_periods_hr:
        if p <= 24:
            risk[p] = "daily_alias"
        elif p <= 720:
            risk[p] = "monthly_alias"
        else:
            risk[p] = "annual_alias"
    return risk


def _build_notes(
    regime, n_obs, n_nights, n_seasons, baseline_days,
    obs_per_night_median, night_duration_hr, snr_proxy,
    lcdb_period, lcdb_u_code, reliability_ceiling,
) -> str:
    """Build a human-readable summary string."""
    parts = [
        f"Regime: {regime}",
        f"N={n_obs} obs across {n_nights} nights",
        f"Baseline={baseline_days:.1f} days",
        f"Median {obs_per_night_median:.0f} obs/night "
        f"over {night_duration_hr:.1f} hr windows",
        f"SNR proxy={snr_proxy:.1f}",
    ]
    if n_seasons > 1:
        parts.append(f"Seasons={n_seasons}")
    if not np.isnan(lcdb_period):
        parts.append(f"LCDB known P={lcdb_period:.3f}hr U={lcdb_u_code}")
    parts.append(f"Reliability ceiling: {reliability_ceiling}")
    return " | ".join(parts)
