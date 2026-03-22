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
    bq_project: str = "lsst-484623"
    bq_dataset: str = "atlast_photometry"
    bq_table:   str = "public_obs_x05"

    @property
    def bq_table_full(self) -> str:
        return f"{self.bq_project}.{self.bq_dataset}.{self.bq_table}"

    # Which bands to use (u/Lu excluded by default — too few observations).
    # Uses canonical long-form names (Lg/Lr/Li) which is what the band_remap
    # below produces. Short names (g/r/i) are remapped before this filter
    # runs, so both naming conventions in source data work transparently.
    bands_use: List[str] = field(default_factory=lambda: ["Lg", "Lr", "Li"])

    # Band remapping: normalises short LSST filter names to canonical long form.
    # Applied in ingestion._post_process BEFORE the bands_use filter, so data
    # files using either convention (g/r/i or Lg/Lr/Li) are handled correctly.
    # Lg=g, Lr=r, Li=i, Lu=u, Lz=z, Ly=y — same physical filter, two naming
    # conventions used by different parts of the Rubin/LSST stack.
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

    # ── ZTF augmentation (Change 7) ───────────────────────────────────────────
    # When use_ztf=True, pipeline.run_single_asteroid() fetches ZTF photometry
    # from IRSA and merges it with Rubin data for objects in the sparse regime.
    # Requires astroquery and internet access. Use False for bulk offline runs.
    #
    # Trigger criteria (all must hold before ZTF is fetched):
    #   - regime == "sparse"  OR  n_obs < ztf_trigger_n_obs
    #   - The object is NOT already in combined or rich_multiyear regime
    #   - At least ztf_min_obs ZTF detections are returned
    #
    # Methodological principle: fetch/no-fetch is decided on data-quality
    # criteria BEFORE running the period search, not after seeing a failure.
    use_ztf:                     bool  = False
    ztf_search_radius_arcsec:    float = 3.0    # cone radius for IRSA queries
    ztf_n_ephemeris_points:      int   = 60     # Horizons probe density
    ztf_time_window_days:        float = 0.4    # ±days around each probe epoch
    ztf_min_obs:                 int   = 15     # minimum ZTF detections to use
    ztf_trigger_n_obs:           int   = 40     # Rubin N_obs threshold to trigger
    ztf_apply_offsets:           bool  = False  # apply ZTF→Rubin zero-point shift
    ztf_date_start:              str   = "2018-03-01"  # ZTF survey start

    # ── ATLAS augmentation (Change 7 Phase 4 — stub) ─────────────────────────
    # ATLAS o/c bands are broadband — treated as independent periodogram check,
    # not merged into the gri multiband fit. Not yet implemented.
    use_atlas: bool = False


# ── Period search ─────────────────────────────────────────────────────────────

@dataclass
class PeriodConfig:
    """Period search grid and algorithm settings."""

    period_min_hr:  float = 0.005 # hard floor (18s) — actual floor computed
                                  # dynamically via Eyer & Bartholdi (1999) in preprocess()
                                  # For Rubin First Look this resolves to ~0.023hr
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
    agreement_tol:      float = 0.10  # 10% — matches Greenstreet et al. 2026
    mhaov_pval_thresh:  float = 0.001
    bayesian_ci_thresh: float = 0.5
    clean_peak_ratio:   float = 3.0

    # ── MBLS false alarm probability (permutation test) ───────────────────────
    # mbls_fap_thresh: FAP threshold for MBLS significance gate.
    #   Matched to mhaov_pval_thresh so both gates use the same alpha.
    # mbls_fap_n_perm: number of permutations.
    #   200 gives FAP resolution of 0.005 — adequate for a 0.001 threshold.
    #   Increase to 1000 for validation runs where exact FAP values matter.
    mbls_fap_thresh:    float = 0.001
    mbls_fap_n_perm:    int   = 200

    # ── Pass 2 expansion gate: GLS FAP threshold (Change 3) ──────────────────
    # Pass 2 of Tier 1 expands the period grid to the Eyer & Bartholdi floor
    # (~0.023hr for Rubin) when the coarse GLS result is not significant.
    # Previously this used a fixed power threshold of 0.15 (arbitrary).
    # Now uses the analytical GLS false alarm probability (Zechmeister &
    # Kürster 2009, A&A 496, 577, Eq. 13; Horne & Baliunas 1986 for M):
    #
    #   FAP = 1 - (1 - (1 - power)^((N-3)/2))^M
    #   M ≈ T_baseline × (f_max - f_min)  [independent frequencies]
    #
    # A coarse FAP above this threshold means the coarse-grid peak is not
    # significant: there is no evidence that the true period lies in the
    # coarse range, so expanding to the fine grid is warranted.
    # 0.05 is the conventional significance level (95% confidence).
    gls_fap_expand_thresh: float = 0.05


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass
class OutputConfig:
    """Catalog and file output settings."""

    results_dir:      str  = "results"
    catalog_file:     str  = "results/period_catalog.csv"
    log_file:         str  = "results/pipeline.log"
    save_plots:       bool = False
    plots_dir:        str  = "results/plots"
    precompute_dir:   str  = ""      # if set, saves .npz arrays per asteroid here
    verbose:          bool = True


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
