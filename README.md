# Asteroid Rotation Period Pipeline

Automated rotation period detection pipeline for LSST/Rubin survey photometry.
Implements a three-tier architecture: fast screening → period refinement → disambiguation.

## Pipeline architecture

```
LSST nightly stream
      │
  Tier 1: GLS + MBLS N=1          ← millions of objects, fast screening
      │
  SNR > 3 and Nobs > 20?
      │ Yes
  Tier 2: MHAOV NH=2              ← thousands of objects, model-based
           MBLS Nterms=2
           Conditional Entropy
      │
  All 3 agree within 5%?
      │ Yes                No → Tier 3: CLEAN alias deconvolution → flag for follow-up
      │
  Publish period to catalog
```

## Methods used

| Tier | Method | Why |
|------|--------|-----|
| 1 | GLS (Generalised Lomb-Scargle) | Fast, statistically correct baseline |
| 1 | MBLS Nterms=1 | Multi-band leverage at low cost |
| 2 | MHAOV NH=2 | Real p-values, models double-hump directly |
| 2 | MBLS Nterms=2 | Independent multi-band confirmation |
| 2 | Conditional Entropy | Model-free validator |
| 3 | CLEAN              | Alias deconvolution from window function |
| 3 | CLEAN | Alias deconvolution from window function |

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
│   ├── tier1.py           # Fast screening: GLS + MBLS
│   ├── tier2.py           # Period refinement: MHAOV + MBLS + CE
│   ├── tier3.py           # Disambiguation: CLEAN alias deconvolution
│   ├── reliability.py     # R-code reliability assessment
│   ├── catalog.py         # Results storage and output
│   ├── window.py          # Spectral window function (alias visualisation)
│   ├── pipeline.py        # Orchestration — runs all tiers
│   └── sources/
│       └── lcdb.py        # LCDB lookup and comparison
├── notebooks/
│   └── pipeline_colab.ipynb   # Main Colab notebook
└── tests/
    └── test_pipeline.py       # Unit tests for each module
```

## Quick start (Google Colab)

1. Open `notebooks/pipeline_colab.ipynb` in Colab
2. Run the setup cell to install dependencies and authenticate BigQuery
3. Set your BigQuery project ID in the config cell
4. Run all cells

## Key parameters (config.py)

- `SNR_THRESHOLD` = 3.0 — minimum amplitude/noise ratio to proceed to Tier 2
- `MIN_OBS` = 20 — minimum observations required
- `PERIOD_MIN_HR` = 0.5 — shortest period to search (hours)
- `PERIOD_MAX_HR` = 24.0 — longest period to search (hours)
- `AGREEMENT_TOL` = 0.05 — fractional tolerance for method agreement (5%)
- `MHAOV_PVAL_THRESH` = 0.001 — significance threshold for period detection
- `BAYESIAN_CI_THRESH` = 0.5 — max 95% CI width (hours) for tentative publish

## Notes on missing fields

Without `heliodist`/`geodist`, the pipeline uses a quadratic polynomial to
detrend the geometry effect. This is a reasonable approximation but will
introduce ~0.05–0.1 mag systematic error in reduced magnitude. Adding distance
columns from MPC/JPL Horizons is strongly recommended for production use.
