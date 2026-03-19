"""
preprocessing.py
----------------
Prepares a single asteroid's observations for period analysis.

Steps
-----
1. Per-band median offset correction  — removes colour differences between bands
2. Geometry detrending                — removes slow brightness trend from
                                        changing Sun/Earth distance
                                        (quadratic polynomial proxy;
                                         replace with HG model if heliodist
                                         and geodist are available)
3. Quality assessment                 — returns SNR, amplitude, per-band counts
4. MJD → hours conversion            — all period methods work in hours

Functions
---------
preprocess(df_obj, config)            — full preprocessing for one asteroid
compute_snr(y_dt, dy)                 — signal-to-noise on detrended LC
quality_report(df_obj, config)        — quick summary without full preprocessing
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

    Attributes
    ----------
    provid      : asteroid provisional designation
    t_hrs       : time array in hours from first observation (N,)
    y_dt        : detrended magnitude offsets (N,) — band offsets removed,
                  geometry trend removed
    dy          : photometric uncertainties (N,)
    bands       : band label per observation (N,)
    df          : original DataFrame with added columns (mag_corr, night)
    n_obs       : total observations
    n_bands     : number of distinct bands used
    baseline_hr : total time span in hours
    amplitude   : peak-to-peak of y_dt (proxy for lightcurve amplitude)
    snr         : amplitude / median(dy)
    band_counts : dict of {band: count}
    poly_coeffs : quadratic detrend polynomial coefficients
    """
    provid:      str
    t_hrs:       np.ndarray
    y_dt:        np.ndarray
    dy:          np.ndarray
    bands:       np.ndarray
    df:          pd.DataFrame
    n_obs:       int
    n_bands:     int
    baseline_hr: float
    amplitude:   float
    snr:         float
    band_counts: Dict[str, int]
    poly_coeffs: np.ndarray


# ── Main function ─────────────────────────────────────────────────────────────

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

    # 1. Per-band median offset correction
    df = _apply_band_offsets(df, config)

    # 2. Build time array in hours
    t0    = df["mjd"].min()
    t_hrs = (df["mjd"].values - t0) * 24.0

    y     = df["mag_corr"].values
    dy    = df["rmsmag"].values

    # 3. Geometry detrend (quadratic polynomial)
    y_dt, poly_coeffs = _geometry_detrend(t_hrs, y)

    # 4. Quality metrics
    amplitude = float(y_dt.max() - y_dt.min())
    snr       = compute_snr(y_dt, dy)

    # 5. Night assignment (useful for plotting)
    df["night"] = (t_hrs / 24).astype(int)

    band_counts = df["band"].value_counts().to_dict()
    n_bands     = len([b for b in band_counts if band_counts[b] >= config.data.min_obs_band])

    logger.debug(
        f"{provid}: N={len(df)}, bands={band_counts}, "
        f"baseline={t_hrs.max():.1f}hr, amp={amplitude:.3f}mag, SNR={snr:.1f}"
    )

    return PreparedData(
        provid      = provid,
        t_hrs       = t_hrs,
        y_dt        = y_dt,
        dy          = dy,
        bands       = df["band"].values,
        df          = df,
        n_obs       = len(df),
        n_bands     = n_bands,
        baseline_hr = float(t_hrs.max()),
        amplitude   = amplitude,
        snr         = snr,
        band_counts = band_counts,
        poly_coeffs = poly_coeffs,
    )


# ── Optional: reduced magnitude (use if heliodist / geodist are available) ────

def apply_reduced_magnitude(
    df: pd.DataFrame,
    heliodist_col: str = "heliodist",
    geodist_col:   str = "geodist",
) -> pd.DataFrame:
    """
    Compute reduced magnitude: H = mag - 5*log10(r * delta)

    This is the physically correct geometry correction, replacing the
    quadratic polynomial proxy in preprocess(). Use when heliodist and
    geodist columns are available.

    Parameters
    ----------
    df : pd.DataFrame with columns mag, heliodist (AU), geodist (AU)

    Returns
    -------
    df with added column 'reduced_mag'
    """
    if heliodist_col not in df.columns or geodist_col not in df.columns:
        raise ValueError(
            f"Columns '{heliodist_col}' and '{geodist_col}' not found. "
            "These are needed for proper reduced magnitude. "
            "Add them to your BigQuery query or use the polynomial detrend."
        )
    r     = df[heliodist_col].values  # heliocentric distance (AU)
    delta = df[geodist_col].values    # geocentric distance (AU)
    df["reduced_mag"] = df["mag"] - 5.0 * np.log10(r * delta)
    logger.info("Applied reduced magnitude correction (HG model geometry)")
    return df


# ── SNR and quality ───────────────────────────────────────────────────────────

def compute_snr(y_dt: np.ndarray, dy: np.ndarray) -> float:
    """
    Signal-to-noise ratio: peak-to-peak amplitude divided by median uncertainty.

    A value above ~3 is the threshold for meaningful period detection.
    Values above 10 typically give clean, unambiguous lightcurves.
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

    Returns dict with: n_obs, n_bands, baseline_days, mag_range,
                       median_rmsmag, passes_tier1
    """
    n_obs        = len(df_obj)
    n_bands      = df_obj["band"].nunique()
    baseline_days = df_obj["mjd"].max() - df_obj["mjd"].min()
    mag_range    = df_obj["mag"].max() - df_obj["mag"].min()
    median_err   = df_obj["rmsmag"].median()
    snr_proxy    = mag_range / median_err if median_err > 0 else 0.0

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

def _apply_band_offsets(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Subtract the per-band median magnitude.

    This removes the colour offset between bands (g is ~0.3 mag brighter
    than r for typical grey asteroids) so all bands can be combined.
    The result is stored in df['mag_corr'].
    """
    df["mag_corr"] = np.nan
    for band in df["band"].unique():
        mask = df["band"] == band
        if mask.sum() >= 2:
            median_mag = df.loc[mask, "mag"].median()
            df.loc[mask, "mag_corr"] = df.loc[mask, "mag"] - median_mag
        else:
            # Too few observations in this band — use zero offset
            df.loc[mask, "mag_corr"] = df.loc[mask, "mag"] - df.loc[mask, "mag"].iloc[0]
    return df


def _geometry_detrend(t_hrs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the slow brightness trend caused by changing geometry
    (heliocentric/geocentric distance variation over the observation window).

    Method: fit a quadratic polynomial to time vs magnitude and subtract it.
    This is a proxy for the proper reduced magnitude correction. Over a
    ~12-night window it captures most of the geometry variation.

    NOTE: Replace this with apply_reduced_magnitude() when distance columns
    are available.

    Returns
    -------
    y_dt : detrended magnitude array
    coeffs : polynomial coefficients (for inspection/logging)
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
