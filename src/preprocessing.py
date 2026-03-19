"""
preprocessing.py
----------------
Prepares a single asteroid's observations for period analysis.

Steps
-----
0. Band name normalization        — merges r/g/i and Lr/Lg/Li duplicates
1. Geometry correction (optional) — fetches r, delta, phase from JPL Horizons;
                                     applies reduced magnitude + HG phase model.
                                     Falls back to quadratic polynomial if
                                     use_geometry=False or Horizons fails.
2. Per-band median offset          — removes colour differences between bands
3. Quality assessment              — SNR, amplitude, per-band counts
4. MJD → hours conversion         — all period methods work in hours
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass

from config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class PreparedData:
    """
    Output of preprocess(). Everything downstream needs is here.
    """
    provid:           str
    t_hrs:            np.ndarray
    y_dt:             np.ndarray
    dy:               np.ndarray
    bands:            np.ndarray
    df:               pd.DataFrame
    n_obs:            int
    n_bands:          int
    baseline_hr:      float
    amplitude:        float
    snr:              float
    band_counts:      Dict[str, int]
    poly_coeffs:      np.ndarray
    geometry_applied: bool          # NEW — True if Horizons data was used


# ── Main function ─────────────────────────────────────────────────────────────

def preprocess(
    df_obj: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> PreparedData:
    provid = df_obj["provid"].iloc[0] if "provid" in df_obj.columns else "unknown"
    df     = df_obj.copy().sort_values("mjd").reset_index(drop=True)

    # ── Step 0: Band name normalization ──────────────────────────────────────
    BAND_REMAP = {'g': 'Lg', 'r': 'Lr', 'i': 'Li', 'z': 'Lz', 'y': 'Ly', 'u': 'Lu'}
    df['band'] = df['band'].replace(BAND_REMAP)

    # ── Step 1: Geometry correction ──────────────────────────────────────────
    geometry_applied = False
    mag_src_col = "mag"   # which column feeds into band offsets

    if config.data.use_geometry:
        try:
            from geometry import fetch_geometry, apply_reduced_magnitude, apply_phase_correction
            df = fetch_geometry(df)
            df = apply_reduced_magnitude(df)
            df = apply_phase_correction(df, G=config.data.hg_slope_G)
            mag_src_col      = "mag_phase_corr"
            geometry_applied = True
            logger.info(f"{provid}: geometry correction applied via JPL Horizons")
        except Exception as e:
            logger.warning(
                f"{provid}: geometry correction failed ({e}) "
                f"— falling back to quadratic detrend"
            )

    # ── Step 2: Per-band median offset correction ─────────────────────────────
    df = _apply_band_offsets(df, config, src_col=mag_src_col)

    # ── Step 3: Build time array in hours ─────────────────────────────────────
    t0    = df["mjd"].min()
    t_hrs = (df["mjd"].values - t0) * 24.0
    y     = df["mag_corr"].values
    dy    = df["rmsmag"].values

    # ── Step 4: Geometry detrend (only if Horizons correction wasn't applied) ─
    if geometry_applied:
        # Physical correction already applied — just mean-centre
        y_dt       = y - np.mean(y)
        poly_coeffs = np.array([0.0, 0.0, float(np.mean(y))])
    else:
        y_dt, poly_coeffs = _geometry_detrend(t_hrs, y)

    # ── Step 5: Quality metrics ───────────────────────────────────────────────
    amplitude = float(y_dt.max() - y_dt.min())
    snr       = compute_snr(y_dt, dy)

    df["night"] = (t_hrs / 24).astype(int)
    band_counts = df["band"].value_counts().to_dict()
    n_bands     = len([b for b in band_counts if band_counts[b] >= config.data.min_obs_band])

    logger.debug(
        f"{provid}: N={len(df)}, bands={band_counts}, "
        f"baseline={t_hrs.max():.1f}hr, amp={amplitude:.3f}mag, SNR={snr:.1f}, "
        f"geom={'horizons' if geometry_applied else 'poly'}"
    )

    return PreparedData(
        provid            = provid,
        t_hrs             = t_hrs,
        y_dt              = y_dt,
        dy                = dy,
        bands             = df["band"].values,
        df                = df,
        n_obs             = len(df),
        n_bands           = n_bands,
        baseline_hr       = float(t_hrs.max()),
        amplitude         = amplitude,
        snr               = snr,
        band_counts       = band_counts,
        poly_coeffs       = poly_coeffs,
        geometry_applied  = geometry_applied,
    )


# ── SNR and quality ───────────────────────────────────────────────────────────

def compute_snr(y_dt: np.ndarray, dy: np.ndarray) -> float:
    amplitude = y_dt.max() - y_dt.min()
    noise     = float(np.median(dy))
    return float(amplitude / noise) if noise > 0 else 0.0


def quality_report(
    df_obj: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Dict:
    n_obs         = len(df_obj)
    n_bands       = df_obj["band"].nunique()
    baseline_days = df_obj["mjd"].max() - df_obj["mjd"].min()
    mag_range     = df_obj["mag"].max() - df_obj["mag"].min()
    median_err    = df_obj["rmsmag"].median()
    snr_proxy     = mag_range / median_err if median_err > 0 else 0.0

    passes = (
        n_obs >= config.tier.min_obs
        and snr_proxy >= config.tier.snr_threshold
    )

    return {
        "n_obs":         n_obs,
        "n_bands":       n_bands,
        "baseline_days": baseline_days,
        "mag_range":     mag_range,
        "median_rmsmag": median_err,
        "snr_proxy":     snr_proxy,
        "passes_tier1":  passes,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_band_offsets(
    df: pd.DataFrame,
    config: PipelineConfig,
    src_col: str = "mag",          # NEW — use mag_phase_corr when geometry applied
) -> pd.DataFrame:
    """
    Subtract the per-band median magnitude to merge multi-band data.
    Reads from src_col, writes to mag_corr.
    """
    df["mag_corr"] = np.nan
    for band in df["band"].unique():
        mask = df["band"] == band
        if mask.sum() >= 2:
            median_mag = df.loc[mask, src_col].median()
            df.loc[mask, "mag_corr"] = df.loc[mask, src_col] - median_mag
        else:
            df.loc[mask, "mag_corr"] = (
                df.loc[mask, src_col] - df.loc[mask, src_col].iloc[0]
            )
    return df


def _geometry_detrend(t_hrs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadratic polynomial proxy for geometry correction.
    Used only when use_geometry=False or Horizons is unavailable.
    """
    if len(t_hrs) < 3:
        return y.copy(), np.array([0.0, 0.0, np.mean(y)])

    try:
        coeffs = np.polyfit(t_hrs, y, deg=2)
        trend  = np.polyval(coeffs, t_hrs)
        y_dt   = y - trend
    except np.linalg.LinAlgError:
        logger.warning("Polynomial detrend failed — using raw magnitudes")
        coeffs = np.array([0.0, 0.0, float(np.mean(y))])
        y_dt   = y - np.mean(y)

    return y_dt, coeffs
