# Asteroid Pipeline — Session State
Last updated: 2026-03-22 (Change 10)

## Repository
https://github.com/wonrobot/asteroid-pipeline  branch: main
Latest commit: 13011d9

## Commit History (this project)
- `1eda6df` Fix KeyError: source column + force-remount Drive
- `52d72af` Fix OSError 107: Drive disconnect
- `91c3a93` Fix stale module cache + improve KeyError reporting
- `502bf63` docs: update validation results post-Change-3
- `a5c8ef7` Add Greenstreet Table 3 additional periods
- `a9cfa68` **Change 8**: MBLS-primary architecture — remove CE voting, fix 2-minima gate
- `eb576af` docs: MBLS FAP coarse/fine grid limitation
- `31dbc5d` Fix OSError 107 on logging close: _ResilientFileHandler
- `5acbc74` Fix Change 8 regression: MBLS sig always publishes
- `dc2d039` docs: update validation results post-Change-8
- `0e0565c` **Change 9**: multi-peak T1→T2 handoff + LRT 2-minima test
- `0ec7ede` Fix Change 9 LRT: correct nested model
- `13011d9` **Change 10**: Bonferroni LRT, spin-barrier prior, T1 Trigger C, T2 top-5 peaks

## Validation Results

### Change 8 baseline (best before Change 9/10)
| Set | Published | Exact | half_P | double_P | disagree |
|-----|-----------|-------|--------|----------|---------|
| Val (37) | 36/37 (97%) | 27 (73%) | 3 | 3 | 4 |
| Blind (39) | 38/39 (97%) | 26 (67%) | 5 | 3 | 5 |

### Change 9 (LRT broken — all objects over-doubled)
Exact dropped to 13/13. 18 false doubles. Do not use.

### Change 10 (current — run in progress at session end)
Early log confirms:
- Bonferroni alpha_eff=6.58e-4 active
- MF76 correctly NOT doubled (F=6.08, p=2.70e-3 ≥ threshold)
- MG17, MH40, MH75 correctly doubled via spin-barrier prior
- ME15 truth period (6.9hr) visible as rank-4 MBLS peak (6.897hr)
- T2 top-5 MBLS peaks stored in catalog
- Full results table pending (run was in progress at ~20% when session ended)

## Architecture (current)

### Tier 1
- Hard gates: n_obs≥20, SNR≥3.0
- Pass 1: coarse GLS 2000pts, 0.5–24hr
- Pass 2 triggers:
  - A: near spin barrier (<2.2hr) — Pravec & Harris 2000
  - B: coarse FAP insignificant — Zechmeister & Kürster 2009
  - C (Change 10): coarse MBLS band support < 0.6 — catches ultrafast aliases (MJ71/MU15 class)
- Outputs: best GLS, best MBLS, mbls_peaks (top-5 window-penalised)

### Tier 2
- Fine grid: 100× oversampling, p_min=min(t1_peaks)/4
- Methods: MHAOV (adaptive NH=2-4), MBLS (Nterms=2), CE (annotation only — Rubin floor too small)
- LRT 2-minima (Change 9+10):
  - Nested F-test H0=nterms harmonics of P, H1=H0+sub-fundamental {cos(πt/P),sin(πt/P)}
  - Bonferroni alpha_eff = 0.05/76 = 6.58e-4 (lrt_n_objects=76 in TierConfig)
  - Spin-barrier prior: if P_raw>2.2hr AND MHAOV confirms P/2 → double unconditionally
- MBLS is primary detector; MHAOV is corroboration
- Decision paths: mbls_confirmed / mbls_sig_only / reject / Tier3
- Outputs: top-5 MBLS peaks+powers stored in catalog (t2_mbls_top_periods, t2_mbls_top_powers)

### Key files
- src/tier1.py — Pass 1/2/3, mbls_peaks field in Tier1Result
- src/tier2.py — LRT, SPIN_BARRIER_HR, Tier2Result with mbls_top_periods/powers
- src/config.py — TierConfig: lrt_n_objects=76, t1_band_support_pass2_thresh=0.6
- src/catalog.py — t2_mbls_top_periods, t2_mbls_top_powers columns
- src/sources/greenstreet2026.py — primary + additional periods, check_against_all_periods()
- src/reliability.py — R-codes, two alias layers

## Known Failure Classes (from Change 9 full run, Change 10 partial)

### True failures (12 objects — unfixable without more data)
- Superfast alias-dominated: MJ71 (0.031hr), MU15 (0.4hr) — Trigger C may help
- Superfast P/2: MG56 (found 0.131hr, truth 0.3hr)
- T1 alias lock (MBLS maximally confident in wrong period):
  MN37, MJ23, MH40, MT24, MN7, ME15, MD38, MP21
  Note: ME15 truth now visible as rank-4 MBLS peak (6.897hr) — diagnostic improvement

### Additional Greenstreet periods found (7 objects)
Not real failures — pipeline found published secondary periods.
MM82, MV31, ML10, MD67, MO39, MK68, MX69

### Symmetric double-hump (irreducible)
MA19: truth=8.9hr, both MBLS and MHAOV agree on 4.438hr.
LRT F=0.03 — no asymmetry. Spin-barrier prior requires MHAOV to find P/2=2.219hr (it doesn't).
Published as half_period. Scientifically honest — data cannot distinguish P from 2P.

## Priority Queue (next session)
1. **Wait for Change 10 full results** — run was in progress, need complete table
2. **Trigger C validation** — did MJ71/MU15 get Trigger C? Check logs for "trigger=C-low_band_support"
3. **MBLS top-5 rank analysis** — now that truth rank is stored, check T1-alias-lock cases
4. **LCDB wiring** — wire lcdb_record into run_single_asteroid() for external cross-validation
5. **Bonferroni for production** — set lrt_n_objects=0 when running >1000 objects (raw alpha=0.05)
6. **Period max 48hr** — change period_max_hr from 24→48 in config.py; no validation basis yet

## Colab Setup
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
_PIPELINE_PKGS = ['pipeline','tier1','tier2','tier3','preprocessing','characterise',
    'reliability','catalog','ingestion','window','config','precompute','geometry','sources']
for _m in list(sys.modules.keys()):
    if any(_m == p or _m.startswith(p+'.') for p in _PIPELINE_PKGS):
        del sys.modules[_m]
import subprocess
subprocess.run(['find', REPO, '-name', '*.pyc', '-delete'], capture_output=True)
sys.path.insert(0, f'{REPO}/src')
print('Ready')
```

## BQ Export
bq-results-20260321-232234-1774135371078
Drive: /content/drive/MyDrive/LSST-Claude/bq-results-20260321-232234-1774135371078/pipeline_results/
