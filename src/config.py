"""
config.py
---------
Central configuration for the asteroid rotation period pipeline.
All tunable parameters live here. Change values here, not in the methods.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ── Data source ───────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """BigQuery and data loading settings."""

    # BigQuery
    bq_project:    str = "lsst-484623"
    bq_dataset:    str = "atlast_photometry"
    bq_table:      str = "public_obs_x05"
    bq_table_full: str = "lsst-484623.atlast_photometry.public_obs_x05"

    # Which bands to use (u excluded by default — too few observations)
    bands_use: List[str] = field(default_factory=lambda: ["g", "r", "i"])

    # Band remapping applied after loading (matches LSST pipeline convention)
    band_remap: dict = field(default_factory=lambda: {
        "g": "Lg", "r": "Lr", "i": "Li", "u": "Lu", "y": "Ly", "z": "Lz"
    })

    # Quality cuts
    rmsmag_max:   float = 0.21   # 99th percentile — reject noisy observations
    min_obs_total: int  = 20     # minimum observations across all bands
    min_obs_band:  int  = 5      # minimum per band to use that band


# ── Period search ─────────────────────────────────────────────────────────────

@dataclass
class PeriodConfig:
    """Period search grid and algorithm settings."""

    period_min_hr:  float = 0.5    # shortest period to test (hours)
    period_max_hr:  float = 24.0   # longest period to test (hours)
    n_grid_coarse:  int   = 8_000  # Tier 1 grid points
    n_grid_fine:    int   = 15_000 # Tier 2/3 grid points
    samples_per_peak: int = 20     # oversampling for astropy autopower

    # Fourier / MHAOV harmonics
    mhaov_nh: int = 2   # number of harmonics for MHAOV (2 = double hump)
    mbls_nterms_t1: int = 1  # Tier 1 MBLS terms
    mbls_nterms_t2: int = 2  # Tier 2 MBLS terms

    # Conditional entropy bins
    ce_n_phase: int = 10
    ce_n_mag:   int = 5

    # CLEAN algorithm
    clean_gain:  float = 0.1
    clean_niter: int   = 200


# ── Tier thresholds ───────────────────────────────────────────────────────────

@dataclass
class TierConfig:
    """Decision thresholds for each pipeline tier."""

    # Tier 1 → Tier 2 gate
    snr_threshold: float = 3.0   # amplitude / median(rmsmag) must exceed this
    min_obs:       int   = 20    # minimum observations to attempt Tier 2

    # Tier 2 → publish / Tier 3 gate
    agreement_tol:      float = 0.05   # fractional period agreement (5%)
    mhaov_pval_thresh:  float = 0.001  # MHAOV F-test p-value threshold

    # Tier 3 → publish tentative / flag for follow-up
    bayesian_ci_thresh: float = 0.5   # max 95% CI width (hours)
    clean_peak_ratio:   float = 3.0   # CLEAN top peak must be this × 2nd peak


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass
class OutputConfig:
    """Catalog and file output settings."""

    results_dir:  str  = "results"
    catalog_file: str  = "results/period_catalog.csv"
    log_file:     str  = "results/pipeline.log"
    save_plots:   bool = False   # set True to save phase-fold plots to disk
    plots_dir:    str  = "results/plots"
    verbose:      bool = True


# ── Master config ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Master config — pass this single object through the whole pipeline."""

    data:   DataConfig   = field(default_factory=DataConfig)
    period: PeriodConfig = field(default_factory=PeriodConfig)
    tier:   TierConfig   = field(default_factory=TierConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# Default instance — import this directly if you don't need customisation
DEFAULT_CONFIG = PipelineConfig()
