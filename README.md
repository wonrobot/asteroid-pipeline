# Asteroid Rotation Period Pipeline

Automated rotation period detection pipeline for LSST/Rubin survey photometry.
Implements a three-tier architecture: fast screening → period refinement → disambiguation.

## Pipeline architecture

```
LSST nightly stream
      │
  Tier 1: GLS + MBLS N=1          ← millions of objects, fast screening
          Window function computed    cadence-specific alias scoring
          Pass 1: coarse grid (≥0.5hr)
          Pass 2: expand to Nyquist floor if power weak or near boundary
          Window-penalised peak selection (replaces raw argmax)
      │
  SNR > 3 and Nobs > 20?
      │ Yes
  Tier 2: MHAOV NH=2-4 (adaptive) ← thousands of objects, model-based
           MBLS Nterms=2             window recomputed on finer grid
           Conditional Entropy       contamination-weighted consensus
           MBLS FAP (permutation)    dual significance gate
      │
  Methods agree within 10%?
  Either gate significant?
      │ Yes                No (sig but disagree) → Tier 3: CLEAN → flag for follow-up
      │
  Reliability assessment
  (fixed alias list + cadence-specific window alias check)
      │
  Publish period to catalog
```

## Methods used

| Tier | Method | Why |
|------|--------|-----|
| 1 | GLS (Generalised Lomb-Scargle) | Fast, statistically correct baseline |
| 1 | MBLS Nterms=1 | Multi-band leverage at low cost |
| 1 | Window function | Cadence-specific alias identification at no extra cost |
| 2 | MHAOV NH=2-4 (adaptive) | Real p-values, models double-hump; harmonic order selected by F-test |
| 2 | MBLS Nterms=2 | Independent multi-band confirmation; 2-minima rule applied |
| 2 | Conditional Entropy | Model-free validator (skipped below 1hr) |
| 2 | MBLS FAP (permutation) | Empirical significance from shuffled null distribution |
| 3 | CLEAN | Alias deconvolution from window function |

## Significance gating (dual gate — Change 2)

Previously MHAOV p-value was the sole gate. MHAOV collapses all bands into a
single series before fitting, discarding inter-band information.

MBLS uses all photometric bands jointly — it has strictly more information.
We now compute an empirical false alarm probability (FAP) for MBLS via
permutation test: shuffle time labels, recompute MBLS on a coarse grid, repeat
200 times. The fraction of permutations exceeding observed power is the FAP.

**Decision logic:**

| MHAOV sig | MBLS sig | Outcome |
|-----------|----------|---------|
| ✓ | ✓ | `both_sig` → full confidence gate, R=3 eligible |
| ✓ | ✗ | `either_sig` → partial confidence, R≤2 |
| ✗ | ✓ | `either_sig` → partial confidence, R≤2 (rescues faint multi-band objects) |
| ✗ | ✗ | Reject unless methods agree; no Tier 3 on pure noise |

This means a faint asteroid with 3 bands × 30 obs each (MHAOV p=0.003, marginal)
but MBLS FAP=0.0002 (significant across all 90 joint observations) now publishes
at R=2 instead of being silently rejected.

## Alias detection (two layers — Change 1)

Every adopted period is checked against two independent alias detectors:

**Layer 1 — Fixed list** (`reliability.py: flag_alias_risk`):
Checks against well-known ground-based aliases: 0.5 day, 1 day, 2 day, 0.5 year, 1 year.

**Layer 2 — Cadence-specific window** (`reliability.py: flag_window_alias`):
Uses the spectral window function computed from this asteroid's actual observation
timestamps. Catches dataset-specific aliases from LSST scheduling gaps, moon
avoidance windows, and weather patterns that the fixed list misses entirely.
Each asteroid has a unique observing history. A period sitting on a window peak
gets `r_flag = "cadence_alias"` and R=-1.

The window function is also used in Tier 1 to penalise GLS peak selection
(window_informed_peaks replaces raw argmax) and in Tier 2 for contamination-weighted
consensus selection.

## Data source

- BigQuery: `lsst-484623.atlast_photometry.public_obs_x05`
- Fields: `provid`, `obstime`, `band`, `mag`, `rmsmag`, `inserted_at`
- Recommended additional fields: `heliodist`, `geodist`, `phase_angle`

## Project structure

```
asteroid_pipeline/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py          # All tunable parameters
│   ├── ingestion.py       # BigQuery + CSV data loading
│   ├── preprocessing.py   # Band offsets, detrending, quality cuts
│   ├── characterise.py    # Data regime classification (dense/sparse/multiyear)
│   ├── geometry.py        # JPL Horizons geometry + HG phase correction
│   ├── tier1.py           # Fast screening: GLS + MBLS + window function
│   ├── tier2.py           # Refinement: MHAOV + MBLS + CE + window + MBLS FAP
│   ├── tier3.py           # Disambiguation: CLEAN alias deconvolution
│   ├── reliability.py     # R-code: fixed alias + cadence window + dual gate
│   ├── catalog.py         # Results storage and output
│   ├── window.py          # Spectral window function (used by tier1, tier2, reliability)
│   ├── pipeline.py        # Orchestration — runs all tiers
│   └── sources/
│       └── lcdb.py        # LCDB lookup and comparison
├── notebooks/
│   └── pipeline_colab.ipynb   # Main Colab notebook
└── tests/
    ├── test_pipeline.py        # Unit tests for each module
    └── test_mbls_regression.py # Regression tests for MBLS data path
```

## Quick start (Google Colab)

1. Open `notebooks/pipeline_colab.ipynb` in Colab
2. Run the setup cell to install dependencies and authenticate BigQuery
3. Set your BigQuery project ID in the config cell
4. Run all cells

## Key parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `snr_threshold` | 3.0 | Minimum amplitude/noise ratio for Tier 1 pass |
| `min_obs` | 20 | Minimum observations required |
| `period_min_hr` | 0.01 | Hard floor; effective floor is data-driven Nyquist (~0.047hr for Rubin) |
| `period_max_hr` | 24.0 | Longest period to search (hours) |
| `agreement_tol` | 0.10 | Fractional tolerance for method agreement (10%) |
| `mhaov_pval_thresh` | 0.001 | MHAOV significance threshold |
| `mbls_fap_thresh` | 0.001 | MBLS FAP threshold (permutation test) |
| `mbls_fap_n_perm` | 200 | Permutations for FAP (increase to 1000 for validation) |
| `clean_peak_ratio` | 3.0 | Min CLEAN peak ratio for tentative publish |

## Reliability (R-codes)

| Code | Meaning | Catalog action |
|------|---------|----------------|
| R=3 | High confidence — all methods agree, both gates significant, dense data | Publish |
| R=2 | Moderate confidence — methods agree, one or both gates significant | Publish with caveat |
| R=1 | Low confidence — two-of-three agreement or Tier 3 tentative | Publish flagged |
| R=0 | No reliable period | Do not publish |
| R=-1 | Alias suspect (fixed list) | Do not publish — confirm independently |
| R=-1 cadence_alias | Alias suspect (window-specific cadence) | Do not publish — confirm independently |

## Changelog

### Change 1 — Wire window function into Tier 1/2 and reliability
*Commits: ae75139, 09315ae*

- **tier1**: window function computed on final period grid at negligible cost;
  `window_informed_peaks` replaces raw argmax for GLS peak selection — candidates
  sitting on cadence alias peaks are penalised before any downstream decisions.
  `Tier1Result` gains `window_power`, `gls_contamination`, `mbls_contamination`.
- **tier2**: window recomputed on finer Tier 2 grid; contamination-weighted
  consensus replaces plain median — prefers the least alias-contaminated estimate
  when methods agree. `Tier2Result` gains per-method and consensus contamination scores.
- **reliability**: two-layer alias detection. Layer 1 (fixed list) unchanged.
  Layer 2 (`flag_window_alias`) uses `t1result.window_power` to check if the
  adopted period sits on a cadence-specific window peak — catches scheduling
  aliases the fixed list misses. R-code capped at 2 for high contamination.
  `r_flag` distinguishes `alias` from `cadence_alias`.
- **catalog**: `window_alias_risk`, `window_alias_note` columns added.

### Change 2 — Dual significance gate: MBLS FAP via permutation test
*Commit: 369bb31*

- **config**: `mbls_fap_thresh=0.001`, `mbls_fap_n_perm=200` added to `TierConfig`.
- **tier2**: `compute_mbls_fap()` added — shuffles time labels 200× against a
  coarse 500-point grid to build the MBLS null distribution empirically.
  `Tier2Result` gains `mbls_fap`, `mbls_sig`, `mhaov_sig`, `both_sig`.
  Decision gate changed from MHAOV-only to dual: `either_sig` needed to publish;
  `both_sig` needed for R=3. Hard reject only when neither gate fires.
- **reliability**: R-code branches on `both_sig`. MBLS-only significant path
  explicitly handled — faint multi-band objects MHAOV would have rejected now
  publish at R=2. Safe defaults ensure backward compatibility.
- **catalog**: `t2_mbls_fap` column added.

### Change 4 — Window-qualified Pass 2 expansion in Tier 1
*Commit: (this commit)*

Previously, Tier 1 expanded its period grid from the coarse 0.5hr floor down
to the Nyquist floor (~0.047hr) whenever coarse GLS power was below 0.15.
This triggered unnecessarily when the low power was caused by a cadence alias
suppressing the real signal — the coarse best period was an alias peak, not
evidence of a fast rotator.

Fix: compute the window function on the coarse grid immediately after Pass 1.
If the coarse best period is window-contaminated (`cont > 0.2`) AND power is
weak, Pass 2 expansion is suppressed — the low power is alias suppression, not
a missing fast rotator. Window-penalised peak selection (already in place from
Change 1) then finds the real signal on the existing coarse grid.

`near_boundary` still always expands — a period at 0.6hr could be a 0.5hr
harmonic, which is a genuine concern independent of alias contamination.

When the grid is not expanded, `window_pow_coarse` is reused directly instead
of recomputing — eliminating a redundant GLS-on-ones call.

- **tier1**: `window_pow_coarse` and `coarse_contaminated` computed after Pass 1;
  `should_expand` replaces the old `if (near_boundary or weak_power)` condition;
  window reused on non-expanded path, recomputed on expanded path.

### Planned

- **Change 3**: Validate 0.5hr and 0.15 power thresholds against LCDB
  known-period asteroids. These are currently hardcoded heuristics.
- **Change 5**: MBLS per-band support weight in `two_of_three` reliability path.

## Notes on missing fields

Without `heliodist`/`geodist`, the pipeline uses a quadratic polynomial to
detrend the geometry effect. This is a reasonable approximation but will
introduce ~0.05–0.1 mag systematic error in reduced magnitude. Adding distance
columns from MPC/JPL Horizons is strongly recommended for production use.
