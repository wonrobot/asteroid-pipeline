"""
geometry.py
-----------
Fetch heliocentric/geocentric distances and phase angles from JPL Horizons.
Merges geometry onto observation dataframes for proper reduced magnitude
and phase angle correction.
"""

import logging
import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons

logger = logging.getLogger(__name__)

# Chunk size — Horizons rejects requests with too many epochs at once
HORIZONS_CHUNK = 50


def fetch_geometry(df_obj: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch heliodist (r), geodist (delta), and phase_angle (alpha)
    from JPL Horizons for all observations of a single asteroid.

    Parameters
    ----------
    df_obj : pd.DataFrame
        Observations for one asteroid. Must have columns: provid, mjd.

    Returns
    -------
    df_obj with added columns: r_au, delta_au, phase_angle_deg
    If Horizons fails, columns are added as NaN and a warning is logged.
    """
    provid = df_obj["provid"].iloc[0]
    epochs = df_obj["mjd"].values
    jd_epochs = (epochs + 2400000.5).tolist()  # MJD → JD

    r_au          = np.full(len(df_obj), np.nan)
    delta_au      = np.full(len(df_obj), np.nan)
    phase_angle   = np.full(len(df_obj), np.nan)

    try:
        # Query in chunks to avoid Horizons limits
        all_rows = []
        for i in range(0, len(jd_epochs), HORIZONS_CHUNK):
            chunk = jd_epochs[i:i + HORIZONS_CHUNK]
            obj = Horizons(
                id=provid,
                location='500',       # geocenter
                epochs=chunk,
                id_type='smallbody'
            )
            eph = obj.ephemerides(quantities='19,20,24')
            all_rows.append(eph['datetime_jd', 'delta', 'r', 'alpha'].to_pandas())

        eph_df = pd.concat(all_rows, ignore_index=True)

        # Match back to original epochs by nearest JD
        for idx, jd in enumerate(jd_epochs):
            nearest = (eph_df['datetime_jd'] - jd).abs().idxmin()
            r_au[idx]        = float(eph_df.loc[nearest, 'r'])
            delta_au[idx]    = float(eph_df.loc[nearest, 'delta'])
            phase_angle[idx] = float(eph_df.loc[nearest, 'alpha'])

        logger.info(
            f"{provid}: geometry fetched — "
            f"r={r_au.mean():.3f}±{r_au.std():.4f} AU, "
            f"delta={delta_au.mean():.3f}±{delta_au.std():.4f} AU, "
            f"phase={phase_angle.mean():.2f}±{phase_angle.std():.2f} deg"
        )

    except Exception as e:
        logger.warning(f"{provid}: Horizons query failed ({e}) — geometry will be NaN")

    df_out = df_obj.copy()
    df_out["r_au"]           = r_au
    df_out["delta_au"]       = delta_au
    df_out["phase_angle_deg"] = phase_angle
    return df_out


def apply_reduced_magnitude(df_obj: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reduced magnitude H from apparent magnitude, r, and delta.

    H = mag - 5*log10(r * delta)

    This removes the brightness trend due to changing distance,
    which otherwise aliases into the lightcurve amplitude.

    Requires columns: mag, r_au, delta_au
    Adds column: mag_reduced
    """
    if "r_au" not in df_obj.columns or df_obj["r_au"].isna().all():
        logger.warning("r_au not available — skipping reduced magnitude correction")
        df_obj["mag_reduced"] = df_obj["mag"]
        return df_obj

    df_obj = df_obj.copy()
    df_obj["mag_reduced"] = (
        df_obj["mag"] - 5.0 * np.log10(df_obj["r_au"] * df_obj["delta_au"])
    )
    logger.debug(
        f"Reduced magnitude applied: "
        f"mean correction = {(df_obj['mag_reduced'] - df_obj['mag']).mean():.3f} mag"
    )
    return df_obj


def apply_phase_correction(df_obj: pd.DataFrame, G: float = 0.15) -> pd.DataFrame:
    """
    Apply HG phase function correction to remove phase angle brightening.

    Uses the standard IAU HG model (Bowell et al. 1989).
    G=0.15 is the default slope parameter for S-type asteroids.
    C-types typically use G=0.10, bright asteroids G=0.25.

    Requires column: phase_angle_deg
    Adds column: mag_phase_corr (mag_reduced corrected for phase)
    """
    if "phase_angle_deg" not in df_obj.columns or df_obj["phase_angle_deg"].isna().all():
        logger.warning("phase_angle_deg not available — skipping phase correction")
        if "mag_reduced" in df_obj.columns:
            df_obj["mag_phase_corr"] = df_obj["mag_reduced"]
        else:
            df_obj["mag_phase_corr"] = df_obj["mag"]
        return df_obj

    df_obj = df_obj.copy()
    alpha_rad = np.radians(df_obj["phase_angle_deg"].values)

    # HG model phi functions
    phi1 = np.exp(-3.33 * np.tan(alpha_rad / 2) ** 0.63)
    phi2 = np.exp(-1.87 * np.tan(alpha_rad / 2) ** 1.22)
    phase_func = -2.5 * np.log10((1 - G) * phi1 + G * phi2)

    src = "mag_reduced" if "mag_reduced" in df_obj.columns else "mag"
    df_obj["mag_phase_corr"] = df_obj[src] - phase_func

    logger.debug(
        f"Phase correction applied (G={G}): "
        f"mean correction = {phase_func.mean():.3f} mag, "
        f"range = [{phase_func.min():.3f}, {phase_func.max():.3f}]"
    )
    return df_obj
