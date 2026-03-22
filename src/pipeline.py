"""
pipeline.py
-----------
Main orchestrator. Runs all three tiers for a list of asteroids.

Handles
-------
- Logging setup
- Per-asteroid try/except (one bad object shouldn't stop the run)
- Progress reporting via tqdm
- Catalog accumulation and periodic saves
- Optional ZTF data augmentation (Change 7)

Usage
-----
    from pipeline import run_pipeline
    from config import PipelineConfig
    from ingestion import load_from_csv

    config = PipelineConfig()
    df     = load_from_csv("rubin_x05_20250226.csv", config)
    catalog = run_pipeline(df, config)

ZTF augmentation (Change 7)
---------------------------
Enable by setting config.data.use_ztf=True. For each asteroid in the
sparse regime (or below ztf_trigger_n_obs), the pipeline fetches ZTF
photometry from IRSA and merges it before running the period search.
Requires internet access and astroquery. Not recommended for bulk runs
of millions of objects — use for targeted analyses of hundreds.

Functions
---------
run_pipeline(df, config)           — run full pipeline on a DataFrame
run_single_asteroid(df_obj, config)— run all tiers for one asteroid
setup_logging(config)              — configure logger
"""

import logging
import os
import traceback
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PipelineConfig, DEFAULT_CONFIG
from ingestion import load_single_object, list_objects
from preprocessing import preprocess
from tier1 import run_tier1
from tier2 import run_tier2
from tier3 import run_tier3
from characterise import characterise, DataCharacterisation
from precompute import save_all as _precompute_save_all
from reliability import compute_reliability
from catalog import (
    init_catalog, result_to_row, append_result,
    save_catalog, catalog_summary
)

logger = logging.getLogger(__name__)


# ── Main pipeline entry point ─────────────────────────────────────────────────


# ── Multiprocessing support ───────────────────────────────────────────────────
# Worker state is stored in globals to avoid pickling df on every call.
_worker_df     = None
_worker_config = None

def _worker_init(df_shared, config_shared):
    """Initialise per-worker globals (called once per worker process)."""
    global _worker_df, _worker_config
    _worker_df     = df_shared
    _worker_config = config_shared

def _worker_call(provid):
    """Process one asteroid — called in worker process."""
    try:
        df_obj = load_single_object(provid, _worker_df)
        row    = run_single_asteroid(df_obj, _worker_config)
        return provid, row, None
    except Exception as e:
        return provid, None, str(e)


def run_pipeline_parallel(
    df:           pd.DataFrame,
    config:       PipelineConfig = DEFAULT_CONFIG,
    provids:      Optional[list] = None,
    save_every_n: int = 100,
    n_workers:    int = 2,
) -> pd.DataFrame:
    """
    Parallel version of run_pipeline using multiprocessing.Pool.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes. Default 2 (Colab free tier).
        Use multiprocessing.cpu_count() to detect available cores.

    Notes
    -----
    Workers share the input dataframe via Pool initializer to avoid
    pickling it on every task call (~1.4GB would be prohibitive).
    Results are collected in the main process and checkpointed every
    save_every_n asteroids.

    ZTF augmentation is NOT recommended with parallel workers — each
    worker would make its own IRSA API calls, potentially hitting rate
    limits. Fetch ZTF data first and merge before calling this function.
    """
    import multiprocessing as mp
    from multiprocessing import Pool

    setup_logging(config)
    catalog = init_catalog(config)

    if provids is None:
        provids = df["provid"].unique().tolist()

    # Resume: skip already-processed objects
    already_done = set(catalog["provid"].tolist()) if len(catalog) > 0 else set()
    if already_done:
        n_before = len(provids)
        provids  = [p for p in provids if p not in already_done]
        logger.info(f"Resuming: skipping {n_before - len(provids)} "
                    f"already-processed objects")

    logger.info(f"Starting parallel pipeline: {len(provids)} asteroids, "
                f"{n_workers} workers")

    n_t1_pass = n_t2_pass = n_t3_tentative = n_followup = n_error = 0
    completed = 0

    with Pool(
        processes   = n_workers,
        initializer = _worker_init,
        initargs    = (df, config),
    ) as pool:
        for provid, row, err in tqdm(
            pool.imap_unordered(_worker_call, provids, chunksize=4),
            total=len(provids), desc=f"Processing ({n_workers} workers)"
        ):
            completed += 1

            if err is not None:
                logger.error(f"Error processing '{provid}': {err}")
                n_error += 1
            else:
                catalog = append_result(catalog, row)
                if row.get("t1_passes"):                      n_t1_pass += 1
                if row.get("t2_passes"):                      n_t2_pass += 1
                if row.get("reliability") == "tentative":     n_t3_tentative += 1
                if row.get("reliability") == "followup_needed": n_followup += 1

            if completed % save_every_n == 0:
                save_catalog(catalog, config)
                logger.info(
                    f"Checkpoint [{completed}/{len(provids)}]: "
                    f"T1={n_t1_pass} T2={n_t2_pass} "
                    f"tentative={n_t3_tentative} errors={n_error}"
                )

    save_catalog(catalog, config)
    logger.info(f"Parallel pipeline complete: {len(provids)} processed, "
                f"{n_error} errors")

    if config.output.verbose:
        print(catalog_summary(catalog).to_string(index=False))

    return catalog

def run_pipeline(
    df:                  pd.DataFrame,
    config:              PipelineConfig = DEFAULT_CONFIG,
    provids:             Optional[list] = None,
    save_every_n:        int = 100,
) -> pd.DataFrame:
    """
    Run the full three-tier pipeline on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by ingestion.load_from_csv / load_from_bigquery
    config : PipelineConfig
    provids : list of str, optional
        Subset of asteroid designations to process.
        If None, processes all unique provids in df.
    save_every_n : int
        Save catalog to disk every N asteroids (checkpoint).

    Returns
    -------
    pd.DataFrame — final catalog
    """
    setup_logging(config)
    catalog = init_catalog(config)

    # Determine which objects to process
    if provids is None:
        provids = df["provid"].unique().tolist()

    logger.info(f"Starting pipeline: {len(provids)} asteroids to process")
    logger.info(f"Config: SNR>{config.tier.snr_threshold}, "
                f"Nobs>{config.tier.min_obs}, "
                f"P=[{config.period.period_min_hr},{config.period.period_max_hr}]hr, "
                f"ZTF={'on' if config.data.use_ztf else 'off'}")

    n_t1_pass = n_t2_pass = n_t3_tentative = n_followup = n_error = 0
    n_ztf_augmented = 0

    for i, provid in enumerate(tqdm(provids, desc="Processing asteroids")):
        try:
            df_obj = load_single_object(provid, df)
            row    = run_single_asteroid(df_obj, config)
            catalog = append_result(catalog, row)

            # Tally outcomes
            if row.get("t1_passes"):     n_t1_pass += 1
            if row.get("t2_passes"):     n_t2_pass += 1
            if row.get("reliability") == "tentative":    n_t3_tentative += 1
            if row.get("reliability") == "followup_needed": n_followup += 1
            if row.get("n_sources", 1) > 1:              n_ztf_augmented += 1

        except KeyError:
            logger.warning(f"Asteroid '{provid}' not found in dataset — skipping")
            n_error += 1
        except Exception as e:
            logger.error(f"Error processing '{provid}': {e}")
            logger.debug(traceback.format_exc())
            n_error += 1

        # Periodic checkpoint save
        if (i + 1) % save_every_n == 0:
            save_catalog(catalog, config)
            logger.info(
                f"Checkpoint [{i+1}/{len(provids)}]: "
                f"T1={n_t1_pass} T2={n_t2_pass} "
                f"tentative={n_t3_tentative} followup={n_followup} "
                f"ZTF_aug={n_ztf_augmented} errors={n_error}"
            )

    # Final save
    save_catalog(catalog, config)

    logger.info("Pipeline complete.")
    logger.info(f"  Processed:    {len(provids)}")
    logger.info(f"  Tier1 passed: {n_t1_pass}")
    logger.info(f"  Tier2 passed: {n_t2_pass}  (confirmed periods)")
    logger.info(f"  Tentative:    {n_t3_tentative}")
    logger.info(f"  Follow-up:    {n_followup}")
    logger.info(f"  ZTF augmented:{n_ztf_augmented}")
    logger.info(f"  Errors:       {n_error}")

    if config.output.verbose:
        print(catalog_summary(catalog).to_string(index=False))

    return catalog


# ── Single-asteroid runner ────────────────────────────────────────────────────

def run_single_asteroid(
    df_obj: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Run all tiers for a single asteroid and return a catalog row dict.

    Change 7 — ZTF augmentation
    ----------------------------
    When config.data.use_ztf=True, this function fetches ZTF photometry
    from IRSA for objects in the sparse regime (or below ztf_trigger_n_obs).
    The decision is made on data-quality criteria BEFORE the period search,
    not after seeing a failed result. If at least ztf_min_obs detections
    are returned, they are merged via merge_with_rubin(), characterise()
    is re-run on the combined dataset, and the period search proceeds on
    the augmented lightcurve.

    Trigger criteria (all must hold to attempt ZTF fetch):
      1. config.data.use_ztf is True
      2. Preliminary characterisation shows regime=="sparse"
         OR n_obs < config.data.ztf_trigger_n_obs
      3. Preliminary characterisation does NOT show regime=="combined"
         (avoid double-fetching for objects already augmented externally)
    """
    provid = df_obj["provid"].iloc[0] if "provid" in df_obj.columns else "unknown"
    logger.debug(f"Processing: {provid}")

    pdir = config.output.precompute_dir  # empty string = disabled

    # ── Data characterisation (preliminary) ───────────────────────────────────
    char = characterise(df_obj)
    logger.debug(
        f"{provid}: regime={char.regime} ceiling={char.reliability_ceiling} "
        f"nights={char.n_nights} baseline={char.baseline_days:.1f}d "
        f"sources={char.n_sources}"
    )

    # ── Ensure source column exists ────────────────────────────────────────────
    # DataFrames built manually (e.g. via direct quality cuts in the validation
    # notebook) bypass _post_process and may lack the "source" column. Add it
    # here so downstream code (characterise, _maybe_augment_ztf) never KeyErrors.
    if "source" not in df_obj.columns:
        df_obj = df_obj.copy()
        df_obj["source"] = "Rubin"

    # ── Change 7: Optional ZTF augmentation ───────────────────────────────────
    # Gate: only trigger when the object would benefit from additional data
    # and we haven't already fetched ZTF data externally.
    df_obj = _maybe_augment_ztf(df_obj, char, config, provid)

    # Re-characterise if augmented (source count may have changed)
    if "source" in df_obj.columns and df_obj["source"].nunique() > char.n_sources:
        char = characterise(df_obj)
        logger.debug(
            f"{provid}: post-ZTF regime={char.regime} "
            f"N={char.n_obs} sources={char.n_sources}"
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    data = preprocess(df_obj, config)

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    t1 = run_tier1(data, config)

    if not t1.passes:
        rel = compute_reliability(char, t1)
        if pdir:
            _precompute_save_all(provid, data, t1, None, None, rel, pdir)
        return result_to_row(data, t1, char=char, rel=rel)

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    t2 = run_tier2(data, t1, config)

    if not t2.to_tier3:
        rel = compute_reliability(char, t1, t2)
        if pdir:
            _precompute_save_all(provid, data, t1, t2, None, rel, pdir)
        return result_to_row(data, t1, t2, char=char, rel=rel)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    t3  = run_tier3(data, t2, config)
    rel = compute_reliability(char, t1, t2, t3)
    if pdir:
        _precompute_save_all(provid, data, t1, t2, t3, rel, pdir)
    return result_to_row(data, t1, t2, t3, char=char, rel=rel)


def _maybe_augment_ztf(
    df_obj: pd.DataFrame,
    char:   DataCharacterisation,
    config: PipelineConfig,
    provid: str,
) -> pd.DataFrame:
    """
    Attempt ZTF augmentation when triggered by config and data quality.

    Returns df_obj unchanged if ZTF is disabled, already combined, or
    the fetch returns insufficient data.

    Trigger criteria:
      1. config.data.use_ztf is True
      2. Current regime is "sparse" OR n_obs < ztf_trigger_n_obs
      3. Not already "combined" (avoids double-augmentation)
      4. At least ztf_min_obs ZTF observations returned

    On any failure (network, Horizons, IRSA) logs a warning and returns
    the original df_obj — the pipeline continues on Rubin-only data.
    """
    if not config.data.use_ztf:
        return df_obj

    # Already multi-source — don't augment again
    if char.n_sources > 1 or char.regime == "combined":
        return df_obj

    # Only augment sparse or low-count objects
    n_obs = char.n_obs
    is_sparse    = char.regime == "sparse"
    is_low_count = n_obs < config.data.ztf_trigger_n_obs

    if not (is_sparse or is_low_count):
        logger.debug(
            f"{provid}: ZTF augmentation skipped "
            f"(regime={char.regime}, n_obs={n_obs} >= {config.data.ztf_trigger_n_obs})"
        )
        return df_obj

    logger.info(
        f"{provid}: attempting ZTF augmentation "
        f"(regime={char.regime}, n_obs={n_obs})"
    )

    try:
        from sources.ztf import fetch_ztf, merge_with_rubin
        df_ztf = fetch_ztf(
            provid,
            config=config,
            date_start=config.data.ztf_date_start,
            time_window_days=config.data.ztf_time_window_days,
            apply_offsets=config.data.ztf_apply_offsets,
        )

        if len(df_ztf) < config.data.ztf_min_obs:
            logger.info(
                f"{provid}: ZTF returned {len(df_ztf)} observations "
                f"(< {config.data.ztf_min_obs} minimum) — using Rubin only"
            )
            return df_obj

        df_combined = merge_with_rubin(df_obj, df_ztf,
                                       apply_offsets=config.data.ztf_apply_offsets)
        logger.info(
            f"{provid}: ZTF augmentation successful — "
            f"{n_obs} Rubin + {len(df_ztf)} ZTF = {len(df_combined)} total"
        )
        return df_combined

    except ImportError as e:
        logger.warning(f"{provid}: ZTF augmentation skipped — {e}")
        return df_obj
    except Exception as e:
        logger.warning(
            f"{provid}: ZTF augmentation failed ({type(e).__name__}: {e}) "
            f"— continuing on Rubin-only data"
        )
        return df_obj


def setup_logging(config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """
    Configure logging to both console and log file.
    Call once at the start of a run.

    If the configured log file path is on a remote mount (e.g. Google Drive)
    that is unavailable, falls back to a local log file at
    /content/pipeline_fallback.log so the run is never blocked by a
    Drive connectivity issue.
    """
    os.makedirs(config.output.results_dir, exist_ok=True)

    level = logging.DEBUG if config.output.verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [logging.StreamHandler()]

    # Try to open the configured log file; fall back to a local path if the
    # Drive mount is unavailable (OSError errno 107 = transport endpoint not
    # connected; errno 5 = I/O error on FUSE).
    log_path = config.output.log_file
    try:
        fh = logging.FileHandler(log_path, mode="a")
        handlers.append(fh)
    except OSError as e:
        fallback = "/content/pipeline_fallback.log"
        try:
            fh = logging.FileHandler(fallback, mode="a")
            handlers.append(fh)
            print(f"[setup_logging] Drive log unavailable ({e}); "
                  f"falling back to {fallback}")
        except OSError:
            print("[setup_logging] Could not open any log file — console only")

    for h in handlers:
        h.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logger.info("Logging initialised")
