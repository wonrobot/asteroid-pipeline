"""
catalog.py
----------
Stores pipeline results to a CSV catalog and manages output.

Each asteroid gets one row in the catalog with all key fields.
The catalog is append-safe — it can be updated nightly.

Functions
---------
init_catalog(config)                   — create or load existing catalog
append_result(catalog_df, result)      — add one asteroid to catalog
save_catalog(catalog_df, config)       — write catalog to disk
load_catalog(config)                   — read existing catalog
result_to_row(t1, t2, t3, data)       — convert pipeline results to dict
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd

from config import PipelineConfig, DEFAULT_CONFIG
from reliability import ReliabilityAssessment
from preprocessing import PreparedData
from tier1 import Tier1Result
from tier2 import Tier2Result
from tier3 import Tier3Result

logger = logging.getLogger(__name__)

# Catalog column order
CATALOG_COLUMNS = [
    "provid", "run_timestamp",
    # Tier 1
    "t1_passes", "t1_gls_period_hr", "t1_gls_power", "t1_snr", "t1_n_obs",
    "t1_reject_reason", "t1_pass2_trigger",
    # Tier 2
    "t2_passes", "t2_to_tier3",
    "t2_mhaov_period_hr", "t2_mbls_period_hr", "t2_ce_period_hr",
    "t2_consensus_period_hr", "t2_F_stat", "t2_p_value", "t2_mbls_fap",
    "t2_mbls_band_support_frac", "t2_mbls_n_bands_supporting",
    "t2_amplitude_mag", "t2_agreement", "t2_period_spread_pct",
    "t2_mbls_top_periods", "t2_mbls_top_powers",
    "t2_lrt_f_stat", "t2_lrt_p_value", "t2_lrt_doubled",
    # Tier 3
    "t3_ran", "t3_publish_tentative", "t3_needs_followup",
    "t3_adopted_period_hr", "t3_clean_period_hr",
    "t3_ci_lo", "t3_ci_hi", "t3_ci_width", "t3_clean_peak_ratio",
    # Final adopted values
    "final_period_hr", "final_period_unc_hr", "reliability",
    # Reliability assessment
    "r_code", "r_flag", "alias_risk", "alias_note",
    "window_alias_risk", "window_alias_note",
    "lcdb_agreement", "lcdb_delta_pct", "reliability_notes",
    "period_exceeds_grid", "n_cycles",
    # Data characterisation
    "regime", "n_nights", "n_seasons", "obs_per_night_median",
    "night_duration_hr", "reliability_ceiling", "recommended_methods",
    "lcdb_period_hr", "lcdb_u_code",
    # Metadata
    "n_bands", "baseline_hr", "bands_used",
]


def init_catalog(config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Load existing catalog or create an empty one.

    Returns
    -------
    pd.DataFrame with CATALOG_COLUMNS
    """
    os.makedirs(config.output.results_dir, exist_ok=True)
    path = config.output.catalog_file

    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info(f"Loaded existing catalog: {len(df)} rows from {path}")
        # Add any new columns that didn't exist in old catalog
        for col in CATALOG_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[CATALOG_COLUMNS]

    logger.info(f"Creating new catalog: {path}")
    return pd.DataFrame(columns=CATALOG_COLUMNS)


def result_to_row(
    data:     PreparedData,
    t1result: Tier1Result,
    t2result: Optional[Tier2Result] = None,
    t3result: Optional[Tier3Result] = None,
    char      = None,   # DataCharacterisation | None
    rel       = None,   # ReliabilityAssessment | None
) -> dict:
    """
    Convert pipeline results for one asteroid into a catalog row dict.
    """
    row = {
        "provid":         data.provid,
        "run_timestamp":  datetime.now(timezone.utc).isoformat(),
        # Tier 1
        "t1_passes":      t1result.passes,
        "t1_gls_period_hr": t1result.best_period_gls,
        "t1_gls_power":   t1result.gls_power_max,
        "t1_snr":         t1result.snr,
        "t1_n_obs":       t1result.n_obs,
        "t1_reject_reason": t1result.reject_reason,
        "t1_pass2_trigger": getattr(t1result, "t1_pass2_trigger", None),
        # Metadata
        "n_bands":        data.n_bands,
        "baseline_hr":    data.baseline_hr,
        "bands_used":     ",".join(sorted(data.band_counts.keys())),
    }

    # Tier 2 fields
    if t2result is not None:
        row.update({
            "t2_passes":            t2result.passes,
            "t2_to_tier3":          t2result.to_tier3,
            "t2_mhaov_period_hr":   t2result.best_period_mhaov,
            "t2_mbls_period_hr":    t2result.best_period_mbls,
            "t2_ce_period_hr":      t2result.best_period_ce,
            "t2_consensus_period_hr": t2result.consensus_period,
            "t2_F_stat":            t2result.F_stat,
            "t2_p_value":           t2result.p_value,
            "t2_mbls_fap":          getattr(t2result, "mbls_fap", np.nan),
            "t2_mbls_band_support_frac":    getattr(t2result, "mbls_band_support_frac", np.nan),
            "t2_mbls_n_bands_supporting":   getattr(t2result, "mbls_n_bands_supporting", np.nan),
            "t2_amplitude_mag":     t2result.amplitude,
            "t2_agreement":         t2result.agreement,
            "t2_period_spread_pct": t2result.period_spread_pct,
            # Top-5 MBLS peaks: stored as pipe-separated strings for CSV compatibility
            "t2_mbls_top_periods":  "|".join(
                f"{p:.4f}" for p in getattr(t2result, "mbls_top_periods", [])
            ),
            "t2_mbls_top_powers":   "|".join(
                f"{pw:.6f}" for pw in getattr(t2result, "mbls_top_powers", [])
            ),
            "t2_lrt_f_stat":   getattr(t2result, "lrt_f_stat",  np.nan),
            "t2_lrt_p_value":  getattr(t2result, "lrt_p_value", np.nan),
            "t2_lrt_doubled":  getattr(t2result, "lrt_doubled",  False),
        })
        # If Tier 2 passed (methods agreed), set final values here
        if t2result.passes:
            row["final_period_hr"]     = t2result.consensus_period
            row["final_period_unc_hr"] = t2result.consensus_period * t2result.period_spread_pct
            # two_of_three agreement → tentative; full agreement → confirmed
            if t2result.agreement == "two_of_three":
                row["reliability"] = "tentative_2of3"
            else:
                row["reliability"] = "confirmed"
    else:
        row.update({k: np.nan for k in CATALOG_COLUMNS
                    if k.startswith("t2_")})

    # Tier 3 fields
    if t3result is not None:
        row.update({
            "t3_ran":               True,
            "t3_publish_tentative": bool(t3result.publish_tentative),
            "t3_needs_followup":    bool(t3result.needs_followup),
            "t3_adopted_period_hr": t3result.best_period_adopted,
            "t3_clean_period_hr":   t3result.best_period_clean,
            "t3_ci_lo":             t3result.ci_lo,
            "t3_ci_hi":             t3result.ci_hi,
            "t3_ci_width":          t3result.ci_width,
            "t3_clean_peak_ratio":  t3result.clean_peak_ratio,
        })
        if t3result.publish_tentative:
            row["final_period_hr"]     = t3result.final_period
            row["final_period_unc_hr"] = t3result.final_period_unc
            row["reliability"]         = "tentative"
        elif t3result.needs_followup:
            row["reliability"]         = "followup_needed"
        row["r_code"]              = rel.r_code if rel else None
        row["adopted_period_hr"]   = rel.period_hr if rel else np.nan
        row["period_unc_hr"]       = rel.period_unc_hr if rel else np.nan
    else:
        row.update({k: np.nan for k in CATALOG_COLUMNS
                    if k.startswith("t3_")})
        row["t3_ran"] = False

    # Flag periods that exceeded the search grid via 2-minima doubling
    final_p   = row.get("final_period_hr", np.nan)
    baseline  = data.baseline_hr
    if not np.isnan(final_p) and not np.isnan(baseline) and baseline > 0:
        row["period_exceeds_grid"] = bool(final_p > 24.0)
        row["n_cycles"] = round(baseline / final_p, 1) if final_p > 0 else np.nan
    else:
        row["period_exceeds_grid"] = False
        row["n_cycles"] = np.nan

    # Ensure reliability is always set
    if row.get("reliability") is None or (isinstance(row.get("reliability"), float) and np.isnan(row.get("reliability", 0.0))):
        if not t1result.passes:
            row["reliability"] = "t1_rejected"
        elif t2result is not None and not t2result.passes and not t2result.to_tier3:
            row["reliability"] = "t2_rejected"

    # Fill any missing fields
    for col in CATALOG_COLUMNS:
        if col not in row:
            row[col] = np.nan

    # ── Reliability assessment fields ────────────────────────────────────────
    if rel is not None:
        row["r_code"]            = rel.r_code
        row["r_flag"]            = rel.r_flag
        row["adopted_period_hr"] = rel.period_hr
        row["period_unc_hr"]     = rel.period_unc_hr
        row["alias_risk"]        = rel.alias_risk
        row["alias_note"]        = rel.alias_note
        row["window_alias_risk"] = rel.window_alias_risk
        row["window_alias_note"] = rel.window_alias_note
        row["lcdb_agreement"]    = rel.lcdb_agreement
        row["lcdb_delta_pct"]    = rel.lcdb_delta_pct
        row["reliability_notes"] = rel.notes
    else:
        row["r_code"]            = None
        row["r_flag"]            = None
        row["adopted_period_hr"] = np.nan
        row["period_unc_hr"]     = np.nan
        row["alias_risk"]        = None
        row["alias_note"]        = None
        row["window_alias_risk"] = None
        row["window_alias_note"] = None
        row["lcdb_agreement"]    = None
        row["lcdb_delta_pct"]    = None
        row["reliability_notes"] = None

    # ── Data characterisation fields ─────────────────────────────────────────
    if char is not None:
        row["regime"]               = char.regime
        row["n_nights"]             = char.n_nights
        row["n_seasons"]            = char.n_seasons
        row["obs_per_night_median"] = char.obs_per_night_median
        row["night_duration_hr"]    = char.night_duration_hr
        row["reliability_ceiling"]  = char.reliability_ceiling
        row["recommended_methods"]  = ",".join(char.recommended_methods)
        row["lcdb_period_hr"]       = char.lcdb_period
        row["lcdb_u_code"]          = char.lcdb_u_code
    else:
        row["regime"]               = "unknown"
        row["n_nights"]             = None
        row["n_seasons"]            = None
        row["obs_per_night_median"] = None
        row["night_duration_hr"]    = None
        row["reliability_ceiling"]  = "unknown"
        row["recommended_methods"]  = None
        row["lcdb_period_hr"]       = None
        row["lcdb_u_code"]          = None

    return row


def append_result(
    catalog_df: pd.DataFrame,
    row:        dict,
) -> pd.DataFrame:
    """
    Append one result row to the catalog DataFrame.
    If the provid already exists, update the row (upsert).
    """
    new_row = pd.DataFrame([{col: row.get(col, np.nan) for col in CATALOG_COLUMNS}])

    if row["provid"] in catalog_df["provid"].values:
        catalog_df = catalog_df[catalog_df["provid"] != row["provid"]].copy()
        logger.debug(f"Updating existing catalog entry for {row['provid']}")

    catalog_df = pd.concat([catalog_df, new_row], ignore_index=True)
    return catalog_df


def save_catalog(
    catalog_df: pd.DataFrame,
    config:     PipelineConfig = DEFAULT_CONFIG,
) -> None:
    """Save catalog DataFrame to CSV.

    If the configured path is on a remote mount that has become unavailable,
    saves a local fallback copy to /content/catalog_fallback.csv so no
    results are lost.
    """
    os.makedirs(config.output.results_dir, exist_ok=True)
    primary = config.output.catalog_file
    try:
        catalog_df.to_csv(primary, index=False)
        logger.info(f"Saved catalog: {len(catalog_df)} rows → {primary}")
    except OSError as e:
        fallback = "/content/catalog_fallback.csv"
        catalog_df.to_csv(fallback, index=False)
        logger.warning(
            f"Drive unavailable ({e}); catalog saved locally → {fallback}"
        )


def load_catalog(config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load catalog from CSV."""
    return pd.read_csv(config.output.catalog_file)


def catalog_summary(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a human-readable summary of pipeline outcomes.
    """
    total      = len(catalog_df)
    t1_pass    = catalog_df["t1_passes"].sum()
    t2_pass    = catalog_df["t2_passes"].sum() if "t2_passes" in catalog_df else 0
    confirmed  = (catalog_df["reliability"] == "confirmed").sum()
    tentative  = (catalog_df["reliability"] == "tentative").sum()
    followup   = (catalog_df["reliability"] == "followup_needed").sum()
    t1_reject  = total - t1_pass

    summary = pd.DataFrame([
        {"Stage": "Total processed",          "Count": total},
        {"Stage": "Tier 1 rejected",          "Count": int(t1_reject)},
        {"Stage": "Tier 1 passed → Tier 2",   "Count": int(t1_pass)},
        {"Stage": "Tier 2 agreed → published","Count": int(t2_pass)},
        {"Stage": "Tier 3 → tentative",       "Count": int(tentative)},
        {"Stage": "Flagged for follow-up",    "Count": int(followup)},
        {"Stage": "Published (confirmed)",    "Count": int(confirmed)},
        {"Stage": "Published (tentative)",    "Count": int(tentative)},
    ])
    return summary
