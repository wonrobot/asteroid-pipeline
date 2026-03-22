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
- `(pending)` **Change 11**: LCDB wiring — lcdb_record threaded through run_single_asteroid(), run_pipeline(), run_pipeline_parallel()

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
- src/pipeline.py — Change 11: _load_lcdb_records(), lcdb_record param in run_single_asteroid()
- src/sources/greenstreet2026.py — primary + additional periods, check_against_all_periods()
- src/sources/lcdb.py — load_lcdb(), lookup(), lookup_batch(), compare_to_lcdb()
- src/reliability.py — R-codes, two alias layers
- tests/test_change11_lcdb.py — 17 tests: load/lookup, wiring, compare_to_lcdb all categories

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
1. **Wait for Change 10 full results** — run was in progress, need complete table. Colab scoring cells ready (see below).
2. **Trigger C validation** — did MJ71/MU15 get Trigger C? Check logs / BQ results for `t1_trigger` column.
3. **MBLS top-5 rank analysis** — `t2_mbls_top_periods` column now in catalog; check alias-lock objects.
4. ~~**LCDB wiring**~~ — **DONE (Change 11)**. `lcdb_record` pre-loaded once per batch run, threaded to `characterise()` and `compute_reliability()` in all code paths. 17 tests pass.
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

## Change 10 Scoring Cells (run in Colab to get full results)
```python
# ── Cell 1: Load Change 10 BQ results ─────────────────────────────────────
import pandas as pd, os, sys, math
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BQ_DIR = '/content/drive/MyDrive/LSST-Claude/bq-results-20260321-232234-1774135371078/pipeline_results/'
csvs = sorted([f for f in os.listdir(BQ_DIR) if f.endswith('.csv')])
print(f"Files: {csvs}")
df = pd.concat([pd.read_csv(os.path.join(BQ_DIR, f)) for f in csvs], ignore_index=True)
df = df.drop_duplicates('provid').reset_index(drop=True)
print(f"Rows: {len(df)}  Cols with mbls_top: {'t2_mbls_top_periods' in df.columns}")

# ── Cell 2: Score vs Greenstreet truth ─────────────────────────────────────
sys.path.insert(0, '/content/asteroid-pipeline/src')
from sources.greenstreet2026 import GROUND_TRUTH, check_against_all_periods

VAL_SET = {  # 37 val objects — update if split list changes
    '2025 MA19','2025 MA45','2025 MA46','2025 MC34','2025 MD38','2025 MD40',
    '2025 MD67','2025 MD76','2025 ME15','2025 ME24','2025 ME68','2025 MF76',
    '2025 MG17','2025 MG56','2025 MH40','2025 MH69','2025 MH75','2025 MH86',
    '2025 MJ13','2025 MJ21','2025 MJ23','2025 MJ30','2025 MJ71','2025 MJ79',
    '2025 MK23','2025 MK41','2025 MK68','2025 MK83','2025 MK88','2025 ML10',
    '2025 ML17','2025 ML35','2025 ML52','2025 ML53','2025 MM37','2025 MM81',
    '2025 MM82',
}

def score_df(df):
    rows = []
    for _, r in df.iterrows():
        pid = r['provid']
        fp  = float(r['final_period_hr']) if pd.notna(r.get('final_period_hr')) else float('nan')
        chk = check_against_all_periods(pid, fp) if not math.isnan(fp) else {'match':'no_period','relation':None,'delta_pct':float('nan')}
        pub = str(r.get('t2_passes','')) == 'True'
        rows.append({'provid': pid, 'final_p': fp,
                     'truth_p': GROUND_TRUTH.get(pid,{}).get('period_hr', float('nan')),
                     'match': chk['match'], 'relation': chk['relation'],
                     'published': pub, 'r_code': r.get('r_code',''),
                     'trigger': r.get('t1_trigger', ''),
                     'mbls_top': r.get('t2_mbls_top_periods', ''),
                     'split': 'val' if pid in VAL_SET else 'blind'})
    return pd.DataFrame(rows)

rdf = score_df(df)
for split in ['val', 'blind', 'all']:
    s = rdf if split == 'all' else rdf[rdf.split == split]
    exact  = ((s.match=='primary') & (s.relation=='exact')).sum()
    half   = (s.relation=='P/2').sum()
    dbl    = (s.relation=='2P').sum()
    pub    = s.published.sum()
    print(f"{split:5} ({len(s):2}): pub={pub}  exact={exact}  half={half}  dbl={dbl}")

# ── Cell 3: Trigger C check ─────────────────────────────────────────────────
tc = rdf[rdf.trigger.str.contains('C-low_band', na=False)]
print(f"\nTrigger C fired: {len(tc)} objects")
print(tc[['provid','final_p','truth_p','match','relation','trigger']].to_string(index=False))
for pid in ['2025 MJ71', '2025 MU15']:
    r = rdf[rdf.provid==pid]
    if len(r): print(f"\n{pid}: trigger={r.iloc[0]['trigger']}  final_p={r.iloc[0]['final_p']:.4f}  match={r.iloc[0]['match']} {r.iloc[0]['relation']}")

# ── Cell 4: MBLS top-5 rank for alias-lock failures ─────────────────────────
ALIAS_LOCK = ['2025 MN37','2025 MJ23','2025 MH40','2025 MT24',
              '2025 MN7','2025 ME15','2025 MD38','2025 MP21']
print(f"\n{'PROVID':<15} {'TRUTH_P':>9} {'RANK':>5}  TOP-5 MBLS PERIODS (hr)")
for pid in ALIAS_LOCK:
    r = rdf[rdf.provid==pid]
    if not len(r): print(f"{pid:<15}  NOT IN RESULTS"); continue
    r = r.iloc[0]
    truth_p = GROUND_TRUTH.get(pid,{}).get('period_hr', float('nan'))
    tops_str = str(r['mbls_top']) if pd.notna(r['mbls_top']) else ''
    tops = [float(x) for x in tops_str.split('|') if x] if tops_str else []
    rank = next((i+1 for i,p in enumerate(tops)
                 if not math.isnan(truth_p) and abs(p-truth_p)/truth_p < 0.10), None)
    tops_fmt = '  '.join(f"{p:.3f}" for p in tops) if tops else '(none)'
    print(f"{pid:<15} {truth_p:>9.4f} {str(rank or '-'):>5}  {tops_fmt}")
```
