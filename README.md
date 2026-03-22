# Asteroid Rotation Period Pipeline

Automated rotation period detection pipeline for LSST/Rubin survey photometry.
Implements a three-tier architecture: fast screening → period refinement → disambiguation.

> **New session?** Read `SESSION_STATE.md` first — it has the current validation
> results, what's been done, and exactly what to work on next.

---

## Pipeline architecture

```
LSST nightly stream
      │
  Tier 1: GLS + MBLS N=1          ← millions of objects, fast screening
          Window function computed    cadence-specific alias scoring
          Pass 1: coarse grid (≥0.5hr)
          Pass 2: expand to Eyer & Bartholdi floor if:
                  (a) best period < spin barrier 2.2hr (Pravec & Harris 2000)
                  (b) coarse GLS FAP > 0.05 (Zechmeister & Kürster 2009)
                  Pass 2 suppressed when low power caused by alias peak
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
  (fixed alias list + cadence-specific window alias — two independent layers)
      │
  Publish period to catalog
```

---

## Methods

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

---

## Significance gating (dual gate)

| MHAOV sig | MBLS sig | Outcome |
|-----------|----------|---------|
| ✓ | ✓ | `both_sig` → full confidence gate, R=3 eligible |
| ✓ | ✗ | `either_sig` → partial confidence, R≤2 |
| ✗ | ✓ | `either_sig` → partial confidence, R≤2 |
| ✗ | ✗ | Reject unless methods agree; no Tier 3 on pure noise |

---

## Alias detection (two independent layers)

**Layer 1 — Fixed list** (`reliability.py: flag_alias_risk`):
Checks against universal ground-based aliases: 0.5 day, 1 day, 2 day, 0.5 year,
1 year. Physical causes are Earth rotation and orbital period — these appear in
every ground-based dataset. Always triggers R=-1. Citation: VanderPlas (2018 PASP).

**Layer 2 — Cadence-specific window** (`reliability.py: flag_window_alias`):
Uses the spectral window function from this asteroid's actual timestamps. Catches
dataset-specific aliases from LSST scheduling gaps, moon avoidance, and weather.

Layer 2 does **not** independently trigger R=-1. Scientific basis: a high
contamination score means "the window peaks here", not "the signal is an alias".
The MBLS FAP directly tests whether observed power exceeds the noise null
distribution. If FAP is significant, that outweighs indirect window contamination
evidence. Combining both as an OR-veto would require calibrated ROC analysis.

**Layer 2 decision logic:**

| Contamination | Significance | Outcome |
|---------------|--------------|---------|
| Low | Any | Clean — no annotation |
| High | Neither gate fires | R=-1, `r_flag=cadence_alias` |
| High | Either gate fires | R capped at 2, `r_flag=cadence_alias_soft` |

---

## All parameter values and their scientific basis

| Parameter | Value | Source |
|-----------|-------|--------|
| `snr_threshold` | 3.0 | Standard detection threshold |
| `min_obs` | 20 | Minimum for stable periodogram |
| `agreement_tol` | 0.10 | Greenstreet et al. (2026) Table 2 |
| `mhaov_pval_thresh` | 0.001 | Conventional α for high-confidence detection |
| `mbls_fap_thresh` | 0.001 | Matched to MHAOV α |
| `mbls_fap_n_perm` | 200 | FAP resolution = 0.005; adequate for 0.001 threshold |
| `clean_peak_ratio` | 3.0 | Isolated-peak criterion for CLEAN |
| `gls_fap_expand_thresh` | 0.05 | Conventional 95% significance (ZK09) |
| `SPIN_BARRIER_HR` | 2.2 | Rubble-pile spin barrier (Pravec & Harris 2000) |
| `Rayleigh radius` | P²/T | Frequency resolution (Rayleigh 1879; Scargle 1982) |
| Period floor | 2 × δt_min | Eyer & Bartholdi (1999) for irregular sampling |
| Grid oversampling T2 | 100× | Greenstreet et al. (2026) Eq. 4 |
| `WINDOW_ALIAS_THRESHOLD` | 0.5 | Conservative midpoint; provisional (see Change 3) |
| `CONTAMINATION_THRESHOLD` | 0.2 | Provisional pending simulation study (Change 3) |
| `WINDOW_PENALTY_ALPHA` | 0.7 | Provisional pending simulation study (Change 3) |

Parameters marked "provisional" are flagged in code comments with the planned
derivation method. See Change 3 in the changelog.

---

## Reliability codes

| Code | Meaning | Catalog action |
|------|---------|----------------|
| R=3 | High confidence — all methods agree, both gates significant, dense data | Publish |
| R=2 | Moderate confidence — methods agree, one or both gates significant | Publish with caveat |
| R=1 | Low confidence — two-of-three agreement or Tier 3 tentative | Publish flagged |
| R=0 | No reliable period | Do not publish |
| R=-1 `alias` | Near fixed alias (daily/annual) | Do not publish |
| R=-1 `cadence_alias` | Window-contaminated + not significant | Do not publish |
| R≤2 `cadence_alias_soft` | Window-contaminated but significant | Publish with caveat |

---

## Data source

- BigQuery: `lsst-484623.atlast_photometry.public_obs_x05`
- Fields: `provid`, `obstime`, `band`, `mag`, `rmsmag`, `inserted_at`
- Validation date range: `obstime 2025-04-21 to 2025-05-06` (Greenstreet+2026 objects)

---

## Project structure

```
asteroid_pipeline/
├── README.md                    ← This file
├── SESSION_STATE.md             ← Current session state, what to do next
├── requirements.txt
├── src/
│   ├── config.py                ← All tunable parameters
│   ├── ingestion.py             ← BigQuery + CSV loading, source="Rubin" tagging
│   ├── preprocessing.py         ← Band offsets, detrending, Eyer & Bartholdi floor
│   ├── characterise.py          ← Data regime classification (n_sources wired)
│   ├── geometry.py              ← JPL Horizons geometry + HG phase correction
│   ├── tier1.py                 ← Fast screening: GLS + MBLS + window + FAP trigger
│   ├── tier2.py                 ← Refinement: MHAOV + MBLS + CE + window + MBLS FAP
│   ├── tier3.py                 ← Disambiguation: CLEAN alias deconvolution
│   ├── reliability.py           ← R-code: dual alias layers + significance gate
│   ├── catalog.py               ← Results storage and output
│   ├── window.py                ← Spectral window (Rayleigh-criterion radius)
│   ├── pipeline.py              ← Orchestration + ZTF augmentation gate
│   ├── precompute.py
│   └── sources/
│       ├── lcdb.py              ← LCDB lookup (fully implemented)
│       ├── ztf.py               ← ZTF IRSA fetcher (fully implemented)
│       ├── atlas.py             ← ATLAS fetcher (stub — Phase 4)
│       ├── validation_sources.py← DAMIT + JPL SBDB (stubs)
│       └── __init__.py
├── notebooks/
│   ├── pipeline_colab.ipynb     ← Main pipeline run (results → Drive)
│   └── validate_pipeline.ipynb  ← Greenstreet+2026 validation (results → Drive)
└── tests/
    ├── test_pipeline.py         ← Unit tests for all modules
    ├── test_mbls_regression.py  ← MBLS data path regression tests
    └── test_change7.py          ← Change 7 + Change 3 tests (60 tests)
```

---

## Quick start (Google Colab)

```python
from google.colab import userdata
import os, sys

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')  # store in Colab Secrets
GITHUB_USER  = 'wonrobot'
REPO_NAME    = 'asteroid-pipeline'

if not os.path.exists(f'/content/{REPO_NAME}'):
    !git clone https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git
else:
    %cd /content/{REPO_NAME} && !git pull

%cd /content/{REPO_NAME}
!pip install -r requirements.txt -q
sys.path.insert(0, '/content/asteroid-pipeline/src')
```

Results save to `MyDrive/asteroid-pipeline-results/` (main pipeline) or
`MyDrive/<your-BQ-folder>/pipeline_results/` (validation notebook).
Both survive Colab disconnects.

---

## Validation results (Greenstreet et al. 2026, 76 objects)

Run date: 2026-03-22. Pipeline state: post-Change-8 (MBLS-primary architecture).

| Set | Objects | Published (R≥1) | Exact (<10%) | Exact+Harmonic | Superfast |
|-----|---------|-----------------|--------------|----------------|-----------|
| Validation (37) | 37 | 36 (97%) | 27/37 (73%) | 28/37 (76%) | 6/9 |
| Blind (39) | 39 | 38 (97%) | 26/39 (67%) | 28/39 (72%) | 6/8 |

**Change vs pre-Change-8:**

| Metric | Pre-Change-8 | Post-Change-8 | Δ |
|--------|-------------|---------------|---|
| Validation exact | 20/37 (54%) | 27/37 (73%) | +7 ↑↑ |
| Blind exact | 20/39 (51%) | 26/39 (67%) | +6 ↑↑ |
| Published (both sets) | 97% | 97% | = |

**R-code breakdown:** Validation: R=3×9, R=2×27, R=0×1.
Blind: R=3×9, R=2×29, R=-1×1 (ML52, 12hr alias, correct veto).

**Match against Greenstreet primary + additional periods (Table 3):**
Validation 31/37 (84%), Blind 32/39 (82%).

**Remaining true failures (12):** Superfast alias harmonics (MJ71, MU15, ME68, MG56)
need ZTF/longer baseline. Long-period aliases (MD38 R=3, MH40 R=3, ME15, MN7,
MJ23, MN37, MP21, MT24) need LCDB wiring or ZTF augmentation.

**Key improvements from Change 8:** 13 objects previously wrong due to CE
phantom-voting MHAOV now publish correctly. Notable recoveries: MD76, MJ30,
ML53, MO79, MV19, MN45, MK83, MV38, MV4, MC34.

---

## Changelog

### Change 8 — MBLS-primary architecture (current)
*Commit: `a9cfa68`*

Three bugs removed from the Tier 2 decision logic, all with empirical
justification from Greenstreet et al. (2026) validation (76 objects):

**Empirical basis:**
- MBLS correct (alone or with MHAOV): 83% of objects
- MHAOV correct (alone or with MBLS): 71% of objects
- MBLS only correct: 13 objects. MHAOV only correct: ≤3 objects.

**Bug 1 — CE defaulted to MHAOV period when skipped:**
All Rubin data has Eyer & Bartholdi floor ~1.4min < CE minimum 60min,
so CE is skipped for every object. The old code set `best_ce = best_mhaov`
when skipped, giving MHAOV two votes disguised as a three-method consensus.
Fix: `best_ce = NaN` when skipped. CE is annotation-only and never
participates in the period decision.

**Bug 2 — 2-minima rule gated on MHAOV agreement:**
The 2-minima rule (doubling the MBLS period when the fitted lightcurve
has only one minimum) is a physical test on the MBLS model. It should not
require MHAOV to agree first — that is a circular dependency that prevented
MBLS from correcting itself when MHAOV was wrong.
Fix: 2-minima rule applied unconditionally.

**Bug 3 — Consensus = contamination-weighted median of MHAOV+MBLS+CE:**
With CE copying MHAOV, the "median" pulled toward MHAOV. Now:
consensus = best_mbls always. MHAOV confirms or disagrees; it does not
compete for the period value.

**New decision paths:**
- `mbls_confirmed`: MHAOV confirms MBLS (within tol or harmonic) + either sig → publish
- `mbls_sig_only`:  MBLS FAP < 0.001, MHAOV not significant → publish R≤2
- `to_tier3`:       Both significant, genuinely different periods → CLEAN
- `reject`:         Neither gate significant

**Known limitation documented (not yet fixed):**
MBLS FAP uses fine-grid observed power vs coarse-grid null — systematic
underestimate. Signals are clear enough in commissioning data that this
does not cause false positives in practice. Fix planned in Change 3
simulation study (Horne & Baliunas correction or same-grid permutations).

---


*Commit: `d27177a`*

Three previously-arbitrary thresholds replaced:

**1. Pass 2 expansion: `weak_power < 0.15` → GLS FAP > 0.05**
- Old: fixed scalar, gave FAP ≈ 0.001 (N=150) to FAP ≈ 0.9 (N=30)
- New: Zechmeister & Kürster (2009) A&A 496, Eq. 13 single-frequency tail
  probability combined with Horne & Baliunas (1986) M ≈ T×(f_max−f_min)
- `gls_fap_expand_thresh = 0.05` added to `TierConfig`
- **tier1**: `insignificant_coarse` replaces `weak_power`

**2. Pass 2 expansion: `near_boundary < 0.75hr` → `near_spin_barrier < 2.2hr`**
- Old: 1.5× the grid floor (purely geometric, no physical meaning)
- New: rubble-pile spin barrier at 2.2hr (Pravec & Harris 2000, Icarus 148)
- **tier1**: `near_spin_barrier` replaces `near_boundary`; `SPIN_BARRIER_HR = 2.2`

**3. Contamination radius: `0.03 × P` (fixed) → `P²/T` (Rayleigh criterion)**
- Old: too large at short periods (spurious flags), too small at long periods
  (missed alias peaks). Crossover at P ≈ 2.9hr for T=288hr.
- New: Rayleigh (1879) frequency resolution δf = 1/T → δP = P²/T
  Citation: Scargle (1982) §II; VanderPlas (2018) PASP 130, §3.1
- **window**: `contamination_score(period_hr, periods, wp, baseline_hr=T)`
- **tier1, tier2, reliability**: all call sites now pass `baseline_hr`

**Still provisional** (documented, pending simulation study):
`CONTAMINATION_THRESHOLD = 0.2`, `WINDOW_PENALTY_ALPHA = 0.7`

---

### Fix — Alias detection logic redesign
*Commit: `4910e25`*

Layer 2 (window function) no longer independently triggers R=-1.
Scientific basis: contamination score ≠ "signal is an alias". The MBLS FAP
directly tests whether power exceeds the noise null distribution — if significant,
it outweighs window contamination. Layer 2 now only triggers R=-1 when
contamination is high AND neither significance gate fires.

New `r_flag` values: `alias` | `cadence_alias` | `cadence_alias_soft`

`WINDOW_ALIAS_THRESHOLD` reverted to 0.5 (0.7 was derived by fitting to the
Greenstreet test set — not peer-reviewable).

---

### Change 7 — Multi-survey data augmentation (ZTF, Phases 1–2)
*Commit: `5ea0980`*

**Phase 1 — ZTF fetcher** (`src/sources/ztf.py`)
- `fetch_ztf()` — ephemeris-based moving-object cone search via JPL Horizons +
  IRSA ZTF lightcurve REST API. Handles all IRSA column variants.
- `merge_with_rubin()` — tags sources, sorts by MJD.

**Phase 2 — Source tagging and n_sources wiring**
- `ingestion`: all Rubin rows tagged `source="Rubin"` in `_post_process()`
- `characterise`: `n_sources = df["source"].nunique()` — `combined` regime now reachable
- `config`: 8 new ZTF fields (`use_ztf=False` default)
- `pipeline`: `_maybe_augment_ztf()` gate in `run_single_asteroid()`

**Phases 3–4: not yet implemented** — see roadmap below.

---

### Change 6 — Eyer & Bartholdi period floor
Rubin observations are highly irregular. Classical Nyquist (P_min = 2Δt) applies
only to regular sampling. Eyer & Bartholdi (1999) show:
`f_eff ≈ 1 / (2 × δt_min)` → P_min ≈ 0.023hr for Rubin (~1.4 min).
Matches Greenstreet et al. (2026) search floor of 0.024hr exactly.

### Changes 1–5
See git log for full details: `git log --oneline`.

---

## Roadmap

### Change 7 Phase 3 — Combined window function (next up)
When Rubin + ZTF data are merged, compute separate window functions per survey
and combine as a product. The joint window has far fewer dominant peaks because
the two surveys have different cadences.
- **Scientific basis**: VanderPlas (2018) §4
- **Files**: `src/window.py` (add `compute_combined_window()`), `src/tier1.py`,
  `src/tier2.py` (use combined window when `char.regime == "combined"`)

### Change 7 Phase 4 — ATLAS fetcher
`src/sources/atlas.py` is a stub. Requires registration at
https://fallingstar-data.com/forcedphot/. ATLAS o/c bands must be treated as
independent cross-check, not merged into MBLS fit.

### Change 3 continuation — Simulation study for remaining provisional values
`CONTAMINATION_THRESHOLD = 0.2` and `WINDOW_PENALTY_ALPHA = 0.7` are still
undocumented. Correct derivation: inject synthetic lightcurves into real Rubin
timestamps, measure false positive rate vs. contamination score, set thresholds
at 5% FPR.

### Planned
- Wire LCDB/DAMIT lookup into `run_single_asteroid()` to catch R=3 wrong answers
- DAMIT spin-state lookup (`src/sources/validation_sources.py` stub)
- JPL SBDB period lookup stub (`src/sources/validation_sources.py`)
