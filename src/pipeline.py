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

Usage
-----
    from pipeline import run_pipeline
    from config import PipelineConfig
    from ingestion import load_from_csv

    config = PipelineConfig()
    df     = load_from_csv("rubin_x05_20250226.csv", config)
    catalog = run_pipeline(df, config)

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
from catalog import (
    init_catalog, result_to_row, append_result,
    save_catalog, catalog_summary
)

logger = logging.getLogger(__name__)


# ── Main pipeline entry point ─────────────────────────────────────────────────

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
                f"P=[{config.period.period_min_hr},{config.period.period_max_hr}]hr")

    n_t1_pass = n_t2_pass = n_t3_tentative = n_followup = n_error = 0

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
                f"tentative={n_t3_tentative} followup={n_followup} errors={n_error}"
            )

    # Final save
    save_catalog(catalog, config)

    logger.info("Pipeline complete.")
    logger.info(f"  Processed:    {len(provids)}")
    logger.info(f"  Tier1 passed: {n_t1_pass}")
    logger.info(f"  Tier2 passed: {n_t2_pass}  (confirmed periods)")
    logger.info(f"  Tentative:    {n_t3_tentative}")
    logger.info(f"  Follow-up:    {n_followup}")
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

    This is the function to call interactively in a notebook when
    you want to inspect one asteroid in detail.

    Parameters
    ----------
    df_obj : pd.DataFrame
        Observations for ONE asteroid (all rows must share the same provid)
    config : PipelineConfig

    Returns
    -------
    dict — catalog row (pass to catalog.append_result)
    """
    provid = df_obj["provid"].iloc[0] if "provid" in df_obj.columns else "unknown"
    logger.debug(f"Processing: {provid}")

    # ── Data characterisation ─────────────────────────────────────────────────
    char = characterise(df_obj)
    logger.debug(
        f"{provid}: regime={char.regime} ceiling={char.reliability_ceiling} "
        f"nights={char.n_nights} baseline={char.baseline_days:.1f}d"
    )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    data = preprocess(df_obj, config)

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    t1 = run_tier1(data, config)

    if not t1.passes:
        return result_to_row(data, t1, char=char)

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    t2 = run_tier2(data, t1, config)

    if t2.passes:
        # Methods agreed — publish confirmed period
        return result_to_row(data, t1, t2, char=char)

    if not t2.to_tier3:
        # Signal not significant and methods disagree — archive
        return result_to_row(data, t1, t2, char=char)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    t3 = run_tier3(data, t2, config)
    return result_to_row(data, t1, t2, t3, char=char)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """
    Configure logging to both console and log file.
    Call once at the start of a run.
    """
    os.makedirs(config.output.results_dir, exist_ok=True)

    level = logging.DEBUG if config.output.verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(config.output.log_file, mode="a"),
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logger.info("Logging initialised")
