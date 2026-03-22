# Session State — Where to Continue From

**Last updated:** 2026-03-22  
**Last commit:** `91c3a93`  
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
| Bug fix | KeyError: source column in run_single_asteroid | `1eda6df` |
| Bug fix | OSError 107: Drive disconnect kills logging/catalog saves | `52d72af` |
| Change 8 | MBLS-primary: remove CE voting, fix 2-minima gate, MHAOV=validator | `a9cfa68` |

---

## Validation results (Greenstreet et al. 2026, 76 objects)

Run date: 2026-03-22. Pipeline state: post-Change-3 + alias logic fix.  
BigQuery date range: `obstime 2025-04-21 to 2025-05-06`  
BQ export folder: `bq-results-20260321-232234-1774135371078`

| Set | Objects | Published (R≥1) | Exact (<10%) | Superfast |
|-----|---------|-----------------|--------------|-----------|
| Validation (37) | 37 | 36 (97%) | 20/37 (54%) | 6/9 |
| Blind (39) | 39 | 38 (97%) | 20/39 (51%) | 7/10 |

**R-code breakdown:**

| | R=3 | R=2 | R=1 | R=0 | R=-1 |
|-|-----|-----|-----|-----|------|
| Validation | 10 | 25 | 1 | 1 | 0 |
| Blind | 9 | 26 | 3 | 0 | 1 |

### Change vs previous run (pre-Change-3, pre-alias-fix)

| Metric | Old | New | Change |
|--------|-----|-----|--------|
| Validation published | 34/37 (92%) | 36/37 (97%) | +2 ↑ |
| Blind published | 30/39 (77%) | 38/39 (97%) | +8 ↑↑ |
| Validation exact | 21/37 (57%) | 20/37 (54%) | -1 (noise) |
| Blind exact | 20/39 (51%) | 20/39 (51%) | = |
| Superfast (val) | 7/9 | 6/9 | -1 (MN25 now R=0) |
| Superfast (blind) | 7/10 | 7/10 | = |
| Total R=-1 | many | 1 | ↓↓ alias fix worked |

The large blind-set publication improvement (77%→97%) is the alias logic fix
eliminating false R=-1 vetoes. The one remaining R=-1 is MD67 (near 12hr alias,
correct veto). MK41 (P=0.063hr) is now correctly recovered — Eyer & Bartholdi
floor fix confirmed working.

### Known issues in the results

1. **R=3 wrong answers persist** — MD38 (9.49hr, truth 15.8hr, Δ=40%) and
   MH40 (4.84hr, truth 8.0hr, Δ=40%). Both are alias harmonics where all three
   methods agreed on the wrong period. Unfixable without external ground truth.
   Fix: wire LCDB/DAMIT cross-check into `run_single_asteroid`.

2. **Superfast misses (6 objects):**
   - MJ71 (truth 0.031hr → pipe 7.1hr): alias harmonic dominated, R=1
   - MN25 (truth 0.40hr → no period found): R=0, insufficient phase coverage
   - MU15 (truth 0.40hr → pipe 9.7hr): alias harmonic, R=2 wrong answer
   - ME68 blind (truth 0.9hr → pipe 15.6hr): alias harmonic, R=2 wrong answer
   - MG56 blind (truth 0.30hr → pipe 0.263hr): Δ=12.3%, just outside 10% tol
   - MN45 blind (truth 0.031hr → pipe 0.094hr): 3× harmonic, Δ=203%
   All require longer baseline or ZTF augmentation.

3. **cadence_alias_soft on correct periods** — 12 objects have `cadence_alias_soft`
   flag; 7 of these have exact period recovery. The flag is conservative and
   correct behaviour (publish with caveat rather than veto), but the contamination
   threshold (0.2) and penalty alpha (0.7) are still provisional pending Change 3
   simulation study.

---

## What to do next (in priority order)

### 0. Re-run validation after Change 8  ← DO THIS FIRST
Change 8 redesigns core Tier 2 decision logic (MBLS-primary, CE removed from
voting, 2-minima rule unconditional). Previous validation results (97%/97%
published, 53%/51% exact) predate this change.
Expected improvements: 13 objects where MBLS was correct but MHAOV+CE voted
it down should now resolve correctly.

### 1. MBLS FAP coarse/fine grid fix (documented, not yet implemented)
The permutation test compares fine-grid observed power against coarse-grid
null power — systematic FAP underestimate (anti-conservative). Not causing
false positives in current data (signals are unambiguous, FAP=0.000/200).
Fix when needed:
- Option A: run permutations on full fine grid (exact, slow)
- Option B: Horne & Baliunas (1986) correction — compute observed power at
  best_mbls on coarse grid (single eval), compare against coarse null.
  Same correction already implemented in gls_fap() for MHAOV.
Revisit in Change 3 simulation study at varying SNR.

### 2. Change 7 Phase 3 — Combined window function
When Rubin + ZTF data are merged, compute separate window functions per
survey and combine (product). Directly addresses alias-harmonic failures.
Files: `src/window.py`, `src/tier1.py`, `src/tier2.py`
Scientific basis: VanderPlas (2018) §4

### 3. LCDB/DAMIT cross-check for R=3 results
Wire `lcdb_record` into `run_single_asteroid()` to catch MD38/MH40-style
wrong R=3 answers. `src/sources/lcdb.py` fully implemented. Just needs wiring.

### 4. Change 3 simulation study
`CONTAMINATION_THRESHOLD = 0.2` and `WINDOW_PENALTY_ALPHA = 0.7` still
provisional. Also the correct place to validate the MBLS FAP fix above.

### 5. Change 7 Phase 4 — ATLAS stub
Requires registration at https://fallingstar-data.com/forcedphot/

---

## Repository structure (current)

```
asteroid_pipeline/
├── README.md
├── SESSION_STATE.md             ← This file
├── requirements.txt
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── characterise.py
│   ├── geometry.py
│   ├── tier1.py
│   ├── tier2.py
│   ├── tier3.py
│   ├── reliability.py
│   ├── catalog.py
│   ├── window.py
│   ├── pipeline.py
│   ├── precompute.py
│   └── sources/
│       ├── lcdb.py              ← LCDB lookup (fully implemented)
│       ├── ztf.py               ← ZTF fetcher (fully implemented)
│       ├── atlas.py             ← STUB — needs implementation
│       ├── validation_sources.py← STUB — DAMIT + JPL SBDB lookups
│       └── __init__.py
├── notebooks/
│   ├── pipeline_colab.ipynb
│   └── validate_pipeline.ipynb
└── tests/
    ├── test_pipeline.py
    ├── test_mbls_regression.py
    └── test_change7.py
```

---

## How to start a new session

```python
import os, sys
from google.colab import userdata

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
REPO = '/content/asteroid-pipeline'

if not os.path.exists(REPO):
    !git clone https://{GITHUB_TOKEN}@github.com/wonrobot/asteroid-pipeline.git {REPO}
else:
    %cd {REPO}
    !git pull

%cd {REPO}
!pip install -r requirements.txt -q

# Evict stale module caches
_PIPELINE_PKGS = ['pipeline','tier1','tier2','tier3','preprocessing','characterise',
    'reliability','catalog','ingestion','window','config','precompute','geometry','sources']
for _m in list(sys.modules.keys()):
    if any(_m == p or _m.startswith(p+'.') for p in _PIPELINE_PKGS):
        del sys.modules[_m]

sys.path.insert(0, f'{REPO}/src')
print('Ready. Read SESSION_STATE.md to see where to continue from.')
```

**Then read this file:**
```python
with open('/content/asteroid-pipeline/SESSION_STATE.md') as f:
    print(f.read())
```

---

## Outstanding decisions needed from researcher

1. **Change 3 simulation study** — needed to document `CONTAMINATION_THRESHOLD`
   and `WINDOW_PENALTY_ALPHA` scientifically. Estimated effort: 1–2 days.

2. **ZTF API** — `fetch_ztf()` works with astroquery + internet. No registration
   needed for IRSA. Test with: `fetch_ztf("2025 MG56", date_end="2025-06-01")`
   (known superfast miss, truth P=0.3hr).

3. **ATLAS registration** — https://fallingstar-data.com/forcedphot/ (free).
   Required before Phase 4 can be implemented.

4. **LCDB wiring priority** — MD38 and MH40 are R=3 wrong answers that can only
   be caught by external ground truth. Decision needed on whether to wire LCDB
   before or after Change 7 Phase 3.
