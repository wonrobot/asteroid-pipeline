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
                                     (only for merged series used by GLS/MHAOV/CE)
3. Quality assessment              — SNR, amplitude, per-band counts
4. MJD → hours conversion         — all period methods work in hours

Two data paths are maintained:
  - Merged (y_dt):      band-offset-corrected + detrended → GLS, MHAOV, CE
  - Multiband (y_multiband): geometry-corrected only, raw band labels → MBLS
    MBLS fits per-band means internally — pre-applying offsets discards the
    inter-band colour information it was designed to exploit.
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

    Two parallel data paths:
    ┌─────────────────────────────────────────────────────────────┐
    │  Merged path  (t_hrs, y_dt, dy, bands)                      │
    │  → band offsets applied, geometry detrended                 │
    │  → used by: GLS, MHAOV, Conditional Entropy                 │
    ├─────────────────────────────────────────────────────────────┤
    │  Multiband path  (t_hrs, y_multiband, dy, bands)            │
    │  → geometry-corrected only, NO band offsets                 │
    │  → used by: MBLS (fits per-band means internally)           │
    └─────────────────────────────────────────────────────────────┘
    t_hrs and dy are shared between both paths.
    """
    provid:           str
    # ── Shared ────────────────────────────────────────────────────
    t_hrs:            np.ndarray   # time in hours from first obs (N,)
    dy:               np.ndarray   # photometric uncertainties (N,)
    bands:            np.ndarray   # band label per observation (N,)
    # ── Merged path (GLS / MHAOV / CE) ───────────────────────────
    y_dt:             np.ndarray   # band-offset-corrected + detrended (N,)
    poly_coeffs:      np.ndarray   # detrend polynomial coefficients
    # ── Multiband path (MBLS) ─────────────────────────────────────
    y_multiband:      np.ndarray   # geometry-corrected mag, no band offsets (N,)
    # ── Metadata ──────────────────────────────────────────────────
    df:               pd.DataFrame
    n_obs:            int
    n_bands:          int
    baseline_hr:      float
    amplitude:        float
    snr:              float
    band_counts:      Dict[str, int]
    geometry_applied: bool          # True if JPL Horizons data was used
    period_min_hr:    float          # data-driven period floor — Eyer & Bartholdi 1999 (hours)


# ── Main function ─────────────────────────────────────────────────────────────

def compute_period_floor(
    t_hrs:    np.ndarray,
    clamp_lo: float = 0.005,   # 18 seconds — hard physical floor
    clamp_hi: float = 0.5,     # 30 minutes — fallback for sparse data
) -> float:
    """
    Data-driven minimum detectable period using the Eyer & Bartholdi (1999)
    effective Nyquist frequency for irregularly sampled data.

    Classical Nyquist (P_min = 2Δt) applies only to REGULAR sampling.
    For irregular data, aliases from evenly-spaced cadence do not stack
    coherently, so signals can be detected at frequencies well above the
    classical Nyquist limit. Eyer & Bartholdi (1999, A&AS 135, 1) show
    that for irregular sampling the effective frequency limit is:

        f_eff ≈ 1 / (2 × δt_min)

    where δt_min is the minimum time separation between observations.
    This is empirically supported by Greenstreet et al. (2026), who detect
    MK41 at P=0.063 hr and MN45/MJ71 at P=0.031 hr using Rubin data with
    ~0.68-minute minimum gaps — consistent with f_eff > 240 cycles/day
    (P_min < 0.1 hr) as stated in their Section 4.1.1.

    Implementation
    --------------
    1. Take the 5th percentile of all intra-visit gaps (gaps < 30 min).
       The 5th percentile is more robust than the strict minimum against
       occasional duplicate timestamps or sub-second rounding.
    2. Apply Eyer & Bartholdi: P_min = 2 × gap_5pct
    3. Clamp to [clamp_lo, clamp_hi].

    For Rubin First Look commissioning data:
        typical min gap ≈ 0.68 min = 0.0113 hr
        → P_min = 2 × 0.0113 = 0.023 hr  (≈1.4 minutes)

    This correctly matches Greenstreet's search floor of 0.024 hr and
    allows detection of MK41 (0.063 hr), MG56 (0.264 hr), and in
    principle MN45/MJ71 (0.031 hr) if phase coverage is sufficient.

    The old compute_nyquist_floor (n_cycles × p10_cadence) was conceptually
    wrong — it applied a regular-sampling concept to irregular data and
    produced a floor of ~0.047 hr, unnecessarily excluding fast rotators.

    Parameters
    ----------
    t_hrs    : observation times in hours (sorted or unsorted)
    clamp_lo : hard minimum floor (default 0.005 hr = 18s)
    clamp_hi : fallback when data is too sparse to compute gap (default 0.5 hr)

    Returns
    -------
    period_min_hr : float
    """
    if len(t_hrs) < 4:
        return clamp_hi

    gaps  = np.diff(np.sort(t_hrs))
    intra = gaps[(gaps > 0) & (gaps < 0.5)]   # intra-visit only (< 30 min)

    if len(intra) < 4:
        return clamp_hi

    # 5th percentile of minimum gaps → robust minimum cadence
    gap_p5    = float(np.percentile(intra, 5))
    floor     = 2.0 * gap_p5   # Eyer & Bartholdi effective Nyquist
    return float(np.clip(floor, clamp_lo, clamp_hi))


# Keep old name as alias for backward compatibility with any external code
def compute_nyquist_floor(t_hrs, n_cycles=4.0, clamp_lo=0.01, clamp_hi=0.5):
    """Deprecated — use compute_period_floor (Eyer & Bartholdi 1999)."""
    return compute_period_floor(t_hrs, clamp_lo=clamp_lo, clamp_hi=clamp_hi)


def preprocess(
    df_obj: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> PreparedData:
    """
    Full preprocessing pipeline for a single asteroid.

    Parameters
    ----------
    df_obj : pd.DataFrame
        Observations for one asteroid (from ingestion.load_single_object).
        Required columns: mjd, band, mag, rmsmag

    config : PipelineConfig

    Returns
    -------
    PreparedData — all arrays ready for period analysis
    """
    provid = df_obj["provid"].iloc[0] if "provid" in df_obj.columns else "unknown"
    df     = df_obj.copy().sort_values("mjd").reset_index(drop=True)

    # ── Step 0: Band name normalization ──────────────────────────────────────
    BAND_REMAP = {'g': 'Lg', 'r': 'Lr', 'i': 'Li', 'z': 'Lz', 'y': 'Ly', 'u': 'Lu'}
    df['band'] = df['band'].replace(BAND_REMAP)

    # ── Step 0b: Per-band 3σ outlier rejection ───────────────────────────────
    # Remove observations more than 3σ from the per-band median magnitude.
    # Uses MAD-based sigma (×1.4826) rather than std — robust because std
    # itself gets inflated by outliers, weakening the 3σ cut.
    # Applied per-band to avoid inter-band colour offsets inflating sigma.
    # Follows Greenstreet et al. (2026) Section 3.1.
    keep = np.ones(len(df), dtype=bool)
    for band in df['band'].unique():
        mask = (df['band'] == band).values
        if mask.sum() < 4:
            continue
        mags   = df.loc[mask, 'mag'].values
        med    = float(np.median(mags))
        mad    = float(np.median(np.abs(mags - med)))
        sigma  = mad * 1.4826
        if sigma > 0:
            keep[mask] = np.abs(mags - med) <= 4.0 * sigma
    n_removed = int((~keep).sum())
    if n_removed > 0:
        logger.debug(f"{provid}: removed {n_removed} outliers (4σ MAD per band)")
    df = df[keep].reset_index(drop=True)

    # ── Step 1: Geometry correction (optional) ────────────────────────────────
    # Applies reduced magnitude + HG phase correction if use_geometry=True.
    # On failure, falls back to the quadratic polynomial detrend below.
    geometry_applied = False
    mag_src_col = "mag"   # column that feeds downstream steps

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

    # ── Step 2: Build time array ──────────────────────────────────────────────
    t0    = df["mjd"].min()
    t_hrs = (df["mjd"].values - t0) * 24.0
    dy    = df["rmsmag"].values

    # ── Step 3: Merged path — band offsets then detrend ───────────────────────
    # GLS, MHAOV, and CE need a single combined series.
    df = _apply_band_offsets(df, config, src_col=mag_src_col)
    y  = df["mag_corr"].values

    if geometry_applied:
        # Physical correction already handles the slow trend — just mean-centre
        y_dt        = y - np.mean(y)
        poly_coeffs = np.array([0.0, 0.0, float(np.mean(y))])
    else:
        y_dt, poly_coeffs = _geometry_detrend(t_hrs, y)

    # ── Step 4: Multiband path — trend removed, band offsets kept ─────────────
    # MBLS fits per-band intercepts internally so colour offsets are fine.
    # But it has no trend model — remove the same slow trend as the merged path.
    # poly_coeffs was fit to band-offset-corrected data (mean ~0), so we must
    # also subtract the global mean of raw mag to centre around zero.
    # Per-band colour offsets (~0.3 mag between g/r/i) are preserved for MBLS.
    global_mean = float(df[mag_src_col].mean())
    trend       = np.polyval(poly_coeffs, t_hrs)
    y_multiband = df[mag_src_col].values.copy() - global_mean - trend

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

    # Data-driven period floor — overrides config minimum if tighter
    nyquist_floor = compute_period_floor(t_hrs)   # Eyer & Bartholdi 1999
    effective_min = max(nyquist_floor, config.period.period_min_hr)

    logger.debug(
        f"{provid}: Period floor (Eyer&Bartholdi)={nyquist_floor*60:.1f}min "
        f"effective_min={effective_min*60:.1f}min"
    )

    return PreparedData(
        provid            = provid,
        t_hrs             = t_hrs,
        dy                = dy,
        bands             = df["band"].values,
        y_dt              = y_dt,
        poly_coeffs       = poly_coeffs,
        y_multiband       = y_multiband,
        df                = df,
        n_obs             = len(df),
        n_bands           = n_bands,
        baseline_hr       = float(t_hrs.max()),
        amplitude         = amplitude,
        snr               = snr,
        band_counts       = band_counts,
        geometry_applied  = geometry_applied,
        period_min_hr     = effective_min,
    )


# ── SNR and quality ───────────────────────────────────────────────────────────

def compute_snr(y_dt: np.ndarray, dy: np.ndarray) -> float:
    """
    Signal-to-noise ratio: peak-to-peak amplitude / median uncertainty.
    Values above ~3 indicate meaningful period detection is possible.
    """
    amplitude = y_dt.max() - y_dt.min()
    noise     = float(np.median(dy))
    return float(amplitude / noise) if noise > 0 else 0.0


def quality_report(
    df_obj: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Dict:
    """
    Quick quality assessment without full preprocessing.
    Used by Tier 1 to decide whether to proceed.
    """
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
    df:      pd.DataFrame,
    config:  PipelineConfig,
    src_col: str = "mag",
) -> pd.DataFrame:
    """
    Subtract the per-band median magnitude so all bands sit at zero.

    This is only applied to the merged series (y_dt) used by GLS/MHAOV/CE.
    MBLS receives y_multiband directly, without this correction.

    Reads from src_col, writes result to mag_corr.
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


def _geometry_detrend(
    t_hrs: np.ndarray,
    y:     np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadratic polynomial proxy for geometry correction.
    Used only when use_geometry=False or Horizons is unavailable.
    Only applied to the merged series — MBLS does not use this.
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
