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
    rmsmag_max:    float = 0.21  # 99th percentile — reject noisy observations
    min_obs_total: int   = 20    # minimum observations across all bands
    min_obs_band:  int   = 5     # minimum per band to use that band

    # ── Geometry correction (JPL Horizons) ────────────────────────────────────
    # When True, fetches r_au, delta_au, phase_angle from JPL Horizons and
    # applies reduced magnitude + HG phase correction instead of the quadratic
    # polynomial detrend. Adds ~1-2s per asteroid — keep False for bulk runs,
    # set True for validation targets or smaller focused runs.
    use_geometry: bool  = False

    # HG phase function slope parameter (Bowell et al. 1989).
    # S-type asteroids (most NEAs): 0.15
    # C-type (dark, carbonaceous):  0.10
    # High-albedo (E-type, V-type): 0.25
    hg_slope_G:   float = 0.15


# ── Period search ─────────────────────────────────────────────────────────────

@dataclass
class PeriodConfig:
    """Period search grid and algorithm settings."""

    period_min_hr:  float = 0.5
    period_max_hr:  float = 24.0
    n_grid_coarse:  int   = 2_000
    n_grid_fine:    int   = 3_000
    n_grid_ce:      int   = 8_000   # CE needs denser grid (histogram method)
    samples_per_peak: int = 20

    mhaov_nh:       int = 2
    mbls_nterms_t1: int = 1
    mbls_nterms_t2: int = 2

    ce_n_phase: int = 10
    ce_n_mag:   int = 5

    clean_gain:  float = 0.1
    clean_niter: int   = 200


# ── Tier thresholds ───────────────────────────────────────────────────────────

@dataclass
class TierConfig:
    """Decision thresholds for each pipeline tier."""

    snr_threshold:      float = 3.0
    min_obs:            int   = 20
    agreement_tol:      float = 0.05
    mhaov_pval_thresh:  float = 0.001
    bayesian_ci_thresh: float = 0.5
    clean_peak_ratio:   float = 3.0


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass
class OutputConfig:
    """Catalog and file output settings."""

    results_dir:  str  = "results"
    catalog_file: str  = "results/period_catalog.csv"
    log_file:     str  = "results/pipeline.log"
    save_plots:   bool = False
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
