# Session State — Where to Continue From

**Last updated:** 2026-03-22  
**Last commit:** `7155e77`  
**Branch:** `main`

---

## What has been built (all committed, all tested)

### Changes 1–6 (pre-session)
| Change | Description | Status |
|--------|-------------|--------|
| 1 | Window function wired into Tier 1/2 and reliability | ✅ Done |
| 2 | Dual significance gate: MBLS FAP via permutation test | ✅ Done |
| 4 | Window-qualified Pass 2 expansion in Tier 1 | ✅ Done |
| 5 | MBLS per-band support in two_of_three reliability path | ✅ Done |
| 6 | Eyer & Bartholdi period floor replaces mis-applied Nyquist | ✅ Done |

### Changes completed this session
| Change | Description | Commit |
|--------|-------------|--------|
| Change 3 | Heuristic thresholds replaced with peer-reviewable derivations | `d27177a` |
| Change 7 Phase 1 | ZTF fetcher (`src/sources/ztf.py`) | `5ea0980` |
| Change 7 Phase 2 | Source tagging + n_sources wiring + combined regime | `5ea0980` |
| Alias logic fix | Window alias no longer independently vetoes — requires significance | `4910e25` |
| Notebook fix | Both notebooks save to Drive (survive Colab disconnects) | `4910e25` + `5ea0980` |

### Change 3 detail (most important for peer review)
Three previously-arbitrary thresholds replaced:

| Old | New | Citation |
|-----|-----|----------|
| `weak_power < 0.15` (fixed scalar) | `gls_fap > 0.05` (ZK09 FAP formula) | Zechmeister & Kürster (2009) A&A 496, Eq. 13 |
| `near_boundary < 0.75hr` (1.5× grid floor) | `near_spin_barrier < 2.2hr` | Pravec & Harris (2000) Icarus 148 |
| `CONTAMINATION_RADIUS = 0.03 × P` (fixed fraction) | `radius = P²/T` (Rayleigh criterion) | Scargle (1982) §II; VanderPlas (2018) PASP 130 |

---

## Validation results (Greenstreet et al. 2026, 76 objects)

Run via `notebooks/validate_pipeline.ipynb`.  
BigQuery date range: `obstime 2025-04-21 to 2025-05-06`  
BQ export folder: `bq-results-20260321-232234-1774135371078`

| Set | Objects | Published | Exact recovery | Superfast |
|-----|---------|-----------|----------------|-----------|
| Validation (37) | 37 | 34 (92%) | 21/37 (57%) | 7/9 |
| Blind (39) | 39 | 30 (77%) | 20/39 (51%) | 7/10 |

### Known issues in the results (not yet fixed)
1. **R=3 wrong answers exist** — MD38 (40% off, R=3), MH40 blind (40% off, R=3).
   All three methods agreed on the wrong period. This is a fundamental limitation
   when all methods share the same alias structure. Fix: ZTF augmentation (Change 7
   Phase 3) and/or LCDB/DAMIT cross-check. No algorithmic pipeline fix is possible.

2. **Superfast misses** — MJ71 (truth 0.031hr → pipe 7.1hr), MU15 (truth 0.4hr → pipe 9.7hr),
   ME68 blind (truth 0.9hr → pipe 15.6hr). The alias harmonic at 7–15hr won because
   it has more phase coverage in 12 days. More observations or ZTF augmentation needed.

3. **Some R=-1 may still be false** — The alias logic fix (commit `4910e25`) changed
   Layer 2 so window contamination alone no longer vetoes. But the fix needs to be
   re-validated on the Greenstreet set by re-running the pipeline. Not yet done.

---

## What to do next (in priority order)

### 1. Re-run validation after Change 3 + alias fix
The current validation results (above) were produced BEFORE Change 3 and the alias
logic fix. The pipeline has changed significantly. Re-run `validate_pipeline.ipynb`
to get updated numbers. Expected improvements:
- False R=-1s for MK23, MO35, ML35, MU59, MA46 should be resolved
- Pass 2 expansion now uses FAP-based trigger — may recover some superfast objects

**How:** Open `validate_pipeline.ipynb` in Colab, `git pull`, run all cells.

### 2. Change 7 Phase 3 — Combined window function
When Rubin + ZTF data are merged, Tier 2 should compute separate window functions
per survey and combine them (product). This dramatically reduces alias power because
the two surveys have different cadences and gaps.

**Files to change:**
- `src/window.py` — add `compute_combined_window(t_hrs_by_source, periods)`
- `src/tier2.py` — when `char.regime == "combined"`, call the combined window
- `src/tier1.py` — same

**Scientific basis:** VanderPlas (2018) §4 — the joint window of two independent
surveys with different cadences suppresses most single-survey aliases.

### 3. Change 7 Phase 4 — ATLAS stub implementation
`src/sources/atlas.py` is a stub. ATLAS is important for bright objects where Rubin
saturates. Requires free registration at https://fallingstar-data.com/forcedphot/
ATLAS o/c bands must be treated as independent cross-check, not merged into MBLS.

### 4. Change 3 continuation — threshold calibration (simulation study)
The current thresholds for `CONTAMINATION_THRESHOLD` (0.2) and `WINDOW_PENALTY_ALPHA`
(0.7) are still undocumented. The README and code both flag these as "provisional,
pending Change 3 simulation study." The correct derivation:
- Simulate synthetic lightcurves with known periods at varying SNR
- Inject them into real Rubin timestamps
- Measure recovery rate as a function of contamination score
- Set thresholds at the contamination score where false positive rate exceeds 5%

### 5. LCDB/DAMIT cross-check for R=3 results
The R=3 wrong answers (MD38, MH40) can only be caught by external ground truth.
`src/sources/lcdb.py` is fully implemented. `src/sources/validation_sources.py`
has stubs for DAMIT and JPL SBDB. The pipeline already calls `compute_reliability`
with an optional `lcdb_record` — just needs to be wired into `run_single_asteroid`.

---

## Repository structure (current)

```
asteroid_pipeline/
├── README.md                    ← Changelog for all changes
├── SESSION_STATE.md             ← This file — where to continue from
├── requirements.txt
├── src/
│   ├── config.py                ← All params incl. ZTF fields, gls_fap_expand_thresh
│   ├── ingestion.py             ← source="Rubin" tagging
│   ├── preprocessing.py         ← Eyer & Bartholdi period floor
│   ├── characterise.py          ← n_sources wired, combined regime active
│   ├── geometry.py              ← JPL Horizons geometry
│   ├── tier1.py                 ← GLS FAP trigger, spin-barrier trigger, Rayleigh radius
│   ├── tier2.py                 ← MHAOV + MBLS + CE + FAP + band support
│   ├── tier3.py                 ← CLEAN disambiguation
│   ├── reliability.py           ← Dual alias layers, window alias now annotation-only
│   ├── catalog.py               ← CSV output
│   ├── window.py                ← Rayleigh-criterion contamination_score
│   ├── pipeline.py              ← ZTF augmentation gate
│   ├── precompute.py
│   └── sources/
│       ├── lcdb.py              ← LCDB lookup (fully implemented)
│       ├── ztf.py               ← ZTF fetcher (fully implemented)
│       ├── atlas.py             ← STUB — needs implementation
│       ├── validation_sources.py← STUB — DAMIT + JPL SBDB lookups
│       └── __init__.py
├── notebooks/
│   ├── pipeline_colab.ipynb     ← Main pipeline (saves to Drive)
│   └── validate_pipeline.ipynb  ← Greenstreet+2026 validation (saves to Drive)
└── tests/
    ├── test_pipeline.py         ← Unit tests for all modules
    ├── test_mbls_regression.py  ← MBLS regression tests
    └── test_change7.py          ← Change 7 + Change 3 tests (46+14 tests)
```

---

## How to start a new session

```python
# Cell 1 — in any new Colab session
from google.colab import userdata
import os, sys

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
GITHUB_USER  = 'wonrobot'
REPO_NAME    = 'asteroid-pipeline'
REPO_URL     = f'https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git'

if not os.path.exists(f'/content/{REPO_NAME}'):
    !git clone {REPO_URL} /content/{REPO_NAME}
else:
    %cd /content/{REPO_NAME} && !git pull

%cd /content/{REPO_NAME}
!pip install -r requirements.txt -q
sys.path.insert(0, '/content/asteroid-pipeline/src')
print('Ready. Read SESSION_STATE.md to see where to continue from.')
```

**Then read this file:**
```python
with open('/content/asteroid-pipeline/SESSION_STATE.md') as f:
    print(f.read())
```

---

## Outstanding decisions needed from researcher

1. **Validation rerun** — should be done before any further algorithmic changes
   so we have a clean baseline for the current codebase.

2. **Simulation study for Change 3** — needed to document `CONTAMINATION_THRESHOLD`
   and `WINDOW_PENALTY_ALPHA` scientifically. Requires synthetic lightcurve generation
   on real Rubin timestamps. Estimated effort: 1–2 days.

3. **ZTF API registration** — `fetch_ztf()` requires internet access and astroquery.
   Works in Colab with `pip install astroquery`. No registration needed for IRSA.
   Test with: `fetch_ztf("2025 MG56", date_end="2025-06-01")` — this is a known
   superfast object the pipeline currently misses.

4. **ATLAS registration** — https://fallingstar-data.com/forcedphot/ (free).
   Required before Phase 4 can be implemented.
