"""
ztf.py
------
Fetch ZTF photometry for asteroid period analysis.

Implements Change 7 Phase 1 from the README roadmap.

Strategy: ephemeris-based moving-object cone search
---------------------------------------------------
The IRSA ZTF lightcurve REST API is designed for static sources.
For moving objects we:
  1. Fetch predicted sky positions from JPL Horizons at a coarse time grid
  2. Issue narrow-radius, time-windowed cone searches at each predicted position
  3. Deduplicate, band-remap, and return a pipeline-standard DataFrame

This approach works for any asteroid (numbered, named, provisional designation)
including recently-discovered objects that do not yet appear in ZTF pre-computed
solar-system object (SSO) catalogs.

Position uncertainty
--------------------
Rubin astrometry is good to ~10–50 mas. A 3 arcsec cone radius generously
covers typical NEA ephemeris uncertainty over the 0.4-day time window — even
at 60 arcsec/hr apparent motion (a fast NEA) the object moves only ~24 arcsec
in 0.4 days, but the ephemeris itself is accurate to <1 arcsec for well-tracked
objects. For poorly-tracked objects (few observations, long arc gap) increase
search_radius_arcsec to 10–30.

ZTF–Rubin calibration
---------------------
ZTF g/r/i and Rubin Lg/Lr/Li share the same filter design but differ by
~0.02–0.05 mag in zero-point. apply_offsets=False (default) leaves the
correction to MBLS, which fits per-band means internally and absorbs small
zero-point differences automatically. Set apply_offsets=True only when
using GLS or MHAOV directly on the merged dataset.

Pipeline integration
--------------------
    from sources.ztf import fetch_ztf, merge_with_rubin

    df_ztf = fetch_ztf("2017 BQ6", date_end="2025-06-01")
    df_combined = merge_with_rubin(df_rubin, df_ztf)
    # df_combined["source"].unique() → ["Rubin", "ZTF"]
    # characterise() sees n_sources=2 → regime="combined"
"""

import logging
import time
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# IRSA ZTF lightcurve REST API
ZTF_IRSA_URL  = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

# ZTF survey start
ZTF_START_DATE = "2018-03-01"
ZTF_START_JD   = 2458178.5  # 2018-03-01 00:00 UTC

# Band mapping: IRSA filtercode → pipeline canonical names
ZTF_BAND_MAP = {"zg": "Lg", "zr": "Lr", "zi": "Li"}

# Approximate ZTF→Rubin zero-point offsets (mag, additive to ZTF mag)
# Positive offset = ZTF is brighter than Rubin in this band.
# Derived from cross-matching studies (Ofek+2012, Bellm+2019, Dekany+2020).
ZTF_RUBIN_OFFSETS = {"Lg": -0.02, "Lr": -0.01, "Li": 0.03}

# Quality cuts on ZTF photometry
ZTF_MAGERR_MIN    = 0.003   # reject saturated epochs
ZTF_MAGERR_MAX    = 0.30    # reject very noisy epochs
ZTF_CATFLAGS_GOOD = 0       # catflags=0 means no known quality issues

# Horizons query chunk size (number of epochs per HTTP call)
_HORIZONS_CHUNK = 50


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ztf(
    provid: str,
    config=None,
    date_start: Optional[str] = None,
    date_end:   Optional[str] = None,
    search_radius_arcsec: float = 3.0,
    n_ephemeris_points:   int   = 60,
    time_window_days:     float = 0.4,
    apply_offsets:        bool  = False,
    api_delay_s:          float = 0.5,
    max_retries:          int   = 3,
    require_catflags:     bool  = True,
) -> pd.DataFrame:
    """
    Fetch ZTF g/r/i photometry for a single moving asteroid.

    Parameters
    ----------
    provid : str
        MPC provisional or permanent designation, e.g. "2025 MA19" or "433".
    config : PipelineConfig, optional
        If supplied, DataConfig.ztf_search_radius_arcsec overrides the
        search_radius_arcsec parameter.
    date_start : str, optional
        ISO date for the earliest ZTF data to fetch.
        Defaults to ZTF survey start (2018-03-01).
    date_end : str, optional
        ISO date for the latest ZTF data. Defaults to today.
    search_radius_arcsec : float
        Cone search radius around each predicted position. 3 arcsec is
        tight enough to reject unrelated sources; for poorly-tracked objects
        (few observations, long arc gap) increase to 10–30.
    n_ephemeris_points : int
        Number of time steps at which to probe for ZTF data. 60 points
        over a 7-year baseline ≈ monthly cadence. For short-arc objects
        (<30 days of Rubin data) reduce to 20.
    time_window_days : float
        Only accept ZTF detections within ±time_window_days of each
        ephemeris probe epoch. 0.4 days = 9.6 hours.
    apply_offsets : bool
        Apply ZTF_RUBIN_OFFSETS to shift ZTF magnitudes toward the Rubin
        photometric system. Default False — MBLS handles this internally.
    api_delay_s : float
        Seconds to pause between successive IRSA calls. Respects IRSA
        rate limits. Default 0.5s.
    max_retries : int
        Maximum retries per IRSA call with exponential backoff.
    require_catflags : bool
        If True, reject ZTF detections where catflags ≠ 0 (known artifacts).

    Returns
    -------
    pd.DataFrame
        Columns: provid, mjd, band, mag, rmsmag, source="ZTF"
        Empty DataFrame if the object has no ZTF data or the API is
        unreachable.

    Raises
    ------
    ImportError
        If astroquery is not installed (needed for Horizons ephemeris).

    Examples
    --------
    >>> df_ztf = fetch_ztf("2017 BQ6", date_end="2025-06-01")
    >>> print(len(df_ztf), df_ztf["band"].value_counts().to_dict())
    342 {'Lr': 156, 'Lg': 108, 'Li': 78}
    """
    # Pull params from config if supplied
    if config is not None and hasattr(config, "data"):
        search_radius_arcsec = getattr(
            config.data, "ztf_search_radius_arcsec", search_radius_arcsec
        )
        n_ephemeris_points = getattr(
            config.data, "ztf_n_ephemeris_points", n_ephemeris_points
        )

    date_start = date_start or ZTF_START_DATE
    date_end   = date_end   or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"{provid}: fetching ZTF ({date_start} → {date_end})")

    # ── Step 1: JPL Horizons ephemeris ────────────────────────────────────────
    try:
        ephemeris = _get_horizons_ephemeris(
            provid, date_start, date_end, n_ephemeris_points
        )
    except ImportError:
        raise
    except Exception as e:
        logger.warning(f"{provid}: Horizons ephemeris failed ({type(e).__name__}: {e})")
        return _empty_df()

    if len(ephemeris) == 0:
        logger.info(f"{provid}: Horizons returned no ephemeris points for this date range")
        return _empty_df()

    logger.debug(
        f"{provid}: {len(ephemeris)} ephemeris points, "
        f"MJD {ephemeris['mjd'].min():.1f}–{ephemeris['mjd'].max():.1f}"
    )

    # ── Step 2: IRSA cone searches at predicted positions ─────────────────────
    all_chunks = []
    n_empty    = 0

    for _, row in ephemeris.iterrows():
        ra_deg  = float(row["ra"])
        dec_deg = float(row["dec"])
        mjd_cen = float(row["mjd"])

        jd_lo = mjd_cen - time_window_days + 2400000.5
        jd_hi = mjd_cen + time_window_days + 2400000.5

        chunk = _query_irsa(
            ra_deg, dec_deg, search_radius_arcsec,
            jd_lo, jd_hi, max_retries=max_retries
        )
        if chunk is not None and len(chunk) > 0:
            all_chunks.append(chunk)
        else:
            n_empty += 1

        if api_delay_s > 0:
            time.sleep(api_delay_s)

    n_raw = sum(len(c) for c in all_chunks)
    logger.info(
        f"{provid}: {len(ephemeris)} IRSA queries → "
        f"{n_raw} raw detections ({n_empty} empty epochs)"
    )

    if not all_chunks:
        logger.info(f"{provid}: no ZTF detections found")
        return _empty_df()

    # ── Step 3: Combine, deduplicate, standardise ──────────────────────────────
    raw = pd.concat(all_chunks, ignore_index=True)

    # Deduplicate: same (oid, mjd) is the same ZTF detection queried twice
    dedup_cols = ["oid", "mjd"] if "oid" in raw.columns else ["mjd"]
    raw = raw.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # Optional catflags quality filter
    if require_catflags and "catflags" in raw.columns:
        before = len(raw)
        raw = raw[pd.to_numeric(raw["catflags"], errors="coerce").fillna(0) == 0]
        n_flagged = before - len(raw)
        if n_flagged:
            logger.debug(f"{provid}: removed {n_flagged} ZTF detections with catflags≠0")

    # ── Step 4: Build standard columns ────────────────────────────────────────
    df = _standardise(raw, provid, apply_offsets)

    # ── Step 5: Quality cuts ──────────────────────────────────────────────────
    df = df[df["rmsmag"].between(ZTF_MAGERR_MIN, ZTF_MAGERR_MAX)].copy()
    df = df[df["band"].isin(ZTF_BAND_MAP.values())].copy()
    df = df.dropna(subset=["mjd", "mag", "rmsmag", "band"]).reset_index(drop=True)
    df = df.sort_values("mjd").reset_index(drop=True)

    logger.info(
        f"{provid}: ZTF fetch complete — {len(df)} observations, "
        f"bands={df['band'].value_counts().to_dict()}, "
        f"baseline={df['mjd'].max()-df['mjd'].min():.1f} days"
    )
    return df


def merge_with_rubin(
    df_rubin:      pd.DataFrame,
    df_ztf:        pd.DataFrame,
    apply_offsets: bool = False,
) -> pd.DataFrame:
    """
    Combine Rubin and ZTF observations into a single source-tagged DataFrame.

    Adds source="Rubin" to all Rubin rows and ensures ZTF rows carry
    source="ZTF". When the combined DataFrame reaches characterise(), it
    will find n_sources=2 and classify the regime as "combined", enabling
    all methods with the highest reliability ceiling.

    Parameters
    ----------
    df_rubin : pd.DataFrame
        Rubin observations from ingestion. Must have columns:
        provid, mjd, band, mag, rmsmag.
    df_ztf : pd.DataFrame
        ZTF observations from fetch_ztf(). May be empty — if so, this
        function returns df_rubin with source="Rubin" added and is a no-op.
    apply_offsets : bool
        Apply ZTF_RUBIN_OFFSETS to ZTF magnitudes. Default False — MBLS
        handles per-band offsets internally.

    Returns
    -------
    pd.DataFrame sorted by mjd with a "source" column.

    Notes
    -----
    Epoch correction (heliocentric distance changes between surveys) is
    not applied here — it is handled by preprocessing.py via geometry.py
    when config.data.use_geometry=True, which is recommended for combined
    regime runs.
    """
    df_r = df_rubin.copy()
    df_r["source"] = "Rubin"

    if df_ztf is None or len(df_ztf) == 0:
        logger.debug("merge_with_rubin: no ZTF data — returning Rubin-only")
        return df_r.sort_values("mjd").reset_index(drop=True)

    df_z = df_ztf.copy()
    if "source" not in df_z.columns:
        df_z["source"] = "ZTF"

    if apply_offsets:
        for band, offset in ZTF_RUBIN_OFFSETS.items():
            df_z.loc[df_z["band"] == band, "mag"] += offset

    combined = pd.concat([df_r, df_z], ignore_index=True)
    combined = combined.sort_values("mjd").reset_index(drop=True)

    n_r = int((combined["source"] == "Rubin").sum())
    n_z = int((combined["source"] == "ZTF").sum())
    logger.info(
        f"merge_with_rubin: {n_r} Rubin + {n_z} ZTF = {len(combined)} observations"
    )
    return combined


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_horizons_ephemeris(
    provid:     str,
    date_start: str,
    date_end:   str,
    n_points:   int,
) -> pd.DataFrame:
    """
    Retrieve predicted RA/Dec from JPL Horizons at evenly-spaced time steps.

    Returns a DataFrame with columns: mjd, ra, dec (degrees).
    Raises ImportError if astroquery is missing.
    """
    try:
        from astroquery.jplhorizons import Horizons
        from astropy.time import Time
    except ImportError:
        raise ImportError(
            "astroquery is required for ZTF ephemeris fetching. "
            "Install with: pip install astroquery astropy"
        )

    t_start = Time(date_start, format="iso", scale="utc").jd
    t_end   = Time(date_end,   format="iso", scale="utc").jd
    t_end   = min(t_end, Time.now().jd)

    if t_end <= t_start:
        logger.debug(f"{provid}: date range {date_start}→{date_end} is empty or inverted")
        return pd.DataFrame(columns=["mjd", "ra", "dec"])

    jd_grid = np.linspace(t_start, t_end, max(n_points, 2))
    rows    = []

    for i in range(0, len(jd_grid), _HORIZONS_CHUNK):
        chunk = jd_grid[i : i + _HORIZONS_CHUNK].tolist()
        try:
            obj = Horizons(id=provid, location="500", epochs=chunk, id_type="smallbody")
            eph = obj.ephemerides(quantities="1")   # RA, Dec only — fastest
            for eph_row in eph:
                rows.append({
                    "mjd": float(eph_row["datetime_jd"]) - 2400000.5,
                    "ra":  float(eph_row["RA"]),
                    "dec": float(eph_row["DEC"]),
                })
        except Exception as e:
            logger.debug(
                f"{provid}: Horizons chunk {i//len(chunk)+1} failed ({e}) — skipping"
            )

    if not rows:
        return pd.DataFrame(columns=["mjd", "ra", "dec"])

    return pd.DataFrame(rows).sort_values("mjd").reset_index(drop=True)


def _query_irsa(
    ra_deg:        float,
    dec_deg:       float,
    radius_arcsec: float,
    jd_lo:         float,
    jd_hi:         float,
    max_retries:   int = 3,
) -> Optional[pd.DataFrame]:
    """
    Query the IRSA ZTF lightcurve API at a fixed sky position and time window.

    Returns a raw DataFrame of detections (columns as returned by IRSA),
    or None if the request fails or returns no data.
    """
    params = {
        "pos":      f"{ra_deg:.6f},{dec_deg:.6f}",
        "radius":   f"{radius_arcsec}arcsec",
        "startJD":  f"{jd_lo:.5f}",
        "endJD":    f"{jd_hi:.5f}",
        "bandname": "zg,zr,zi",
        "fmt":      "csv",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(ZTF_IRSA_URL, params=params, timeout=30)

            if resp.status_code != 200:
                logger.debug(f"IRSA returned HTTP {resp.status_code} — retry {attempt+1}")
                time.sleep(2.0 ** attempt)
                continue

            text = resp.text.strip()
            if not text:
                return None   # empty = no detections at this position/time

            # IRSA prepends comment lines with '#'; strip them before parsing
            data_lines = [l for l in text.splitlines()
                          if l.strip() and not l.startswith("#")]
            if len(data_lines) < 2:  # header only
                return None

            df = pd.read_csv(StringIO("\n".join(data_lines)))
            return df if len(df) > 0 else None

        except requests.RequestException as e:
            logger.debug(f"IRSA request error ({e}) — retry {attempt+1}")
            time.sleep(2.0 ** attempt)

    logger.warning(f"IRSA query failed after {max_retries} retries at RA={ra_deg:.4f}")
    return None


def _standardise(
    df:            pd.DataFrame,
    provid:        str,
    apply_offsets: bool,
) -> pd.DataFrame:
    """
    Map raw IRSA ZTF columns to pipeline-standard column names.

    IRSA column names vary slightly across ZTF data releases — we handle
    the common variants for each field.
    """
    out = pd.DataFrame()

    # ── Time ──────────────────────────────────────────────────────────────────
    # IRSA returns "mjd" (MJD directly) or "jd" (Julian Date)
    if "mjd" in df.columns:
        out["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")
    elif "jd" in df.columns:
        out["mjd"] = pd.to_numeric(df["jd"], errors="coerce") - 2400000.5
    else:
        logger.warning("ZTF response has no mjd/jd column — skipping chunk")
        return _empty_df()

    # ── Magnitude ─────────────────────────────────────────────────────────────
    for col in ("mag", "psfmag", "medianmag", "mrad"):
        if col in df.columns:
            out["mag"] = pd.to_numeric(df[col], errors="coerce")
            break
    else:
        return _empty_df()

    # ── Uncertainty ───────────────────────────────────────────────────────────
    for col in ("magerr", "psfmagerr", "magerr_auto", "sigmapsf"):
        if col in df.columns:
            out["rmsmag"] = pd.to_numeric(df[col], errors="coerce")
            break
    else:
        out["rmsmag"] = 0.05   # conservative fallback

    # ── Band ──────────────────────────────────────────────────────────────────
    for col in ("filtercode", "bandname", "filter", "fid"):
        if col in df.columns:
            raw_band = df[col].astype(str).str.strip().str.lower()
            # fid is an integer in some releases: 1=zg, 2=zr, 3=zi
            if col == "fid":
                fid_map = {"1": "zg", "2": "zr", "3": "zi"}
                raw_band = raw_band.map(fid_map).fillna("unknown")
            out["band"] = raw_band.map(ZTF_BAND_MAP)
            break
    else:
        out["band"] = "Lr"   # fallback

    # ── Passthrough columns (for deduplication upstream) ──────────────────────
    if "oid" in df.columns:
        out["oid"] = df["oid"]
    if "catflags" in df.columns:
        out["catflags"] = pd.to_numeric(df["catflags"], errors="coerce").fillna(0)

    out["provid"] = provid
    out["source"] = "ZTF"

    if apply_offsets:
        for band, offset in ZTF_RUBIN_OFFSETS.items():
            mask = out["band"] == band
            out.loc[mask, "mag"] = out.loc[mask, "mag"] + offset

    return out[
        [c for c in ["provid", "mjd", "band", "mag", "rmsmag", "source",
                     "oid", "catflags"] if c in out.columns]
    ].copy()


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard column schema."""
    return pd.DataFrame(
        columns=["provid", "mjd", "band", "mag", "rmsmag", "source"]
    )
