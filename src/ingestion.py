"""
ingestion.py
------------
Data loading from BigQuery or CSV fallback.
All I/O is isolated here — no other module reads data directly.

Functions
---------
load_from_bigquery(provids, config)   — fetch one or more asteroids from BQ
load_from_csv(path, config)           — load from local CSV file
load_single_object(provid, df)        — extract one asteroid from a DataFrame
list_objects(df)                      — summary of what's in a DataFrame
"""

import logging
from typing import List, Optional, Union
import pandas as pd
import numpy as np

from config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ── BigQuery loading ──────────────────────────────────────────────────────────

def load_from_bigquery(
    provids: Optional[Union[str, List[str]]] = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    limit: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Load photometry from BigQuery.

    Parameters
    ----------
    provids : str or list of str, optional
        Specific asteroid provisional designations to load.
        If None, loads all objects (use limit to cap rows).
    config : PipelineConfig
    limit : int, optional
        Maximum rows to return. Useful for testing.
    date_start, date_end : str, optional
        ISO date strings e.g. '2025-01-01' to filter by obstime.

    Returns
    -------
    pd.DataFrame with columns: provid, obstime, band, mag, rmsmag, mjd
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError(
            "google-cloud-bigquery not installed. "
            "Run: pip install google-cloud-bigquery db-dtypes pyarrow"
        )

    client = bigquery.Client(project=config.data.bq_project)
    table  = config.data.bq_table_full

    # Build WHERE clauses
    conditions = []

    if provids is not None:
        if isinstance(provids, str):
            provids = [provids]
        ids = ", ".join(f"'{p}'" for p in provids)
        conditions.append(f"provid IN ({ids})")

    if date_start:
        conditions.append(f"obstime >= '{date_start}'")
    if date_end:
        conditions.append(f"obstime <= '{date_end}'")

    conditions.append("provid IS NOT NULL")
    where = f"WHERE {' AND '.join(conditions)}"
    lim   = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
            provid,
            obstime,
            band,
            mag,
            rmsmag,
            UNIX_SECONDS(obstime) / 86400.0 + 40587.0 AS mjd
        FROM `{table}`
        {where}
        ORDER BY provid, obstime
        {lim}
    """

    logger.info(f"Querying BigQuery: {table}")
    logger.debug(f"Query:\n{query}")

    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} rows for {df['provid'].nunique():,} objects")

    return _post_process(df, config)


def load_from_csv(
    path: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load photometry from a local CSV file.

    Expected columns: provid, obstime, band, mag, rmsmag
    Optional:         mjd (will be computed from obstime if missing)

    Parameters
    ----------
    path : str
        Path to CSV file.
    config : PipelineConfig

    Returns
    -------
    pd.DataFrame with standardised columns
    """
    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, parse_dates=["obstime"])

    # Compute MJD from obstime if not present
    if "mjd" not in df.columns:
        df["mjd"] = _obstime_to_mjd(df["obstime"])
        logger.debug("Computed MJD from obstime")

    logger.info(f"Loaded {len(df):,} rows for {df['provid'].nunique():,} objects")
    return _post_process(df, config)


# ── Object extraction ─────────────────────────────────────────────────────────

def load_single_object(
    provid: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract all observations for a single asteroid from a loaded DataFrame.

    Parameters
    ----------
    provid : str
        Provisional designation e.g. '2025 MG21'
    df : pd.DataFrame
        Full dataset as returned by load_from_bigquery or load_from_csv

    Returns
    -------
    pd.DataFrame sorted by MJD, or raises KeyError if not found
    """
    subset = df[df["provid"] == provid].copy().sort_values("mjd").reset_index(drop=True)
    if len(subset) == 0:
        raise KeyError(f"Asteroid '{provid}' not found in dataset")
    logger.debug(f"{provid}: {len(subset)} observations, bands: {subset['band'].value_counts().to_dict()}")
    return subset


def list_objects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of all objects in the dataset.

    Columns: provid, n_obs, n_bands, bands, mjd_min, mjd_max,
             baseline_days, mag_range, median_rmsmag
    """
    def summarise(g):
        return pd.Series({
            "n_obs":         len(g),
            "n_bands":       g["band"].nunique(),
            "bands":         ",".join(sorted(g["band"].unique())),
            "mjd_min":       g["mjd"].min(),
            "mjd_max":       g["mjd"].max(),
            "baseline_days": g["mjd"].max() - g["mjd"].min(),
            "mag_range":     g["mag"].max() - g["mag"].min(),
            "median_rmsmag": g["rmsmag"].median(),
        })

    summary = df.groupby("provid").apply(summarise).reset_index()
    summary = summary.sort_values("n_obs", ascending=False).reset_index(drop=True)
    logger.info(f"Dataset summary: {len(summary):,} unique objects")
    return summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _obstime_to_mjd(obstime: pd.Series) -> pd.Series:
    """Convert pandas datetime series to MJD (Modified Julian Date)."""
    epoch = pd.Timestamp("1858-11-17")
    return (pd.to_datetime(obstime, utc=True).dt.tz_localize(None) - epoch).dt.total_seconds() / 86400.0


def _post_process(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Standardise column types, apply band filter, and sort.
    Called by both BigQuery and CSV loaders.
    """
    # Ensure MJD is present
    if "mjd" not in df.columns:
        df["mjd"] = _obstime_to_mjd(df["obstime"])

    # Cast types
    df["mag"]    = pd.to_numeric(df["mag"],    errors="coerce")
    df["rmsmag"] = pd.to_numeric(df["rmsmag"], errors="coerce")
    df["mjd"]    = pd.to_numeric(df["mjd"],    errors="coerce")

    # Drop rows with null essentials
    before = len(df)
    df = df.dropna(subset=["mag", "rmsmag", "mjd", "band", "provid"])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with null values")

    # Filter to usable bands
    df = df[df["band"].isin(config.data.bands_use)].copy()

    # Apply quality cut on rmsmag
    df = df[df["rmsmag"] <= config.data.rmsmag_max].copy()
    df = df[df["rmsmag"] > 0].copy()

    df = df.sort_values(["provid", "mjd"]).reset_index(drop=True)
    return df
