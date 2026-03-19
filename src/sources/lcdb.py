"""
lcdb.py
-------
Interface to the LCDB (Asteroid Lightcurve Database, Warner et al.)

Download source: https://minplanobs.org/MPInfo/datazips/LCLIST_PUB_CURRENT.zip
File used: lc_summary_pub.txt — one record per asteroid

Key columns in lc_summary_pub.txt:
  NUMBER  : MPC number
  NAME    : name if assigned
  DESIG   : provisional designation (packed MPC format)
  CLASS   : taxonomic class
  H       : absolute magnitude
  G       : HG phase slope parameter
  PERIOD  : rotation period (hours)
  AMIN    : minimum lightcurve amplitude (mag)
  AMAX    : maximum lightcurve amplitude (mag)
  U       : reliability code (1=poor, 2=fair, 3=good)
  BIN     : binary flag

LCDB U codes:
  3   : well-determined — use as ground truth
  2   : likely correct — treat as fair prior
  1   : tentative — treat with caution
  3-  : may be wrong despite high confidence
  2+  : between 2 and 3

Note: DESIG uses packed MPC format. We handle both packed and
unpacked formats in matching. LCDB updates lag new discoveries
by months to years, so recent objects will return found=False.
"""

import logging
import os
import requests
import zipfile
import io
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LCDB_URL      = "https://minplanobs.org/MPInfo/datazips/LCLIST_PUB_CURRENT.zip"
LCDB_FILENAME = "lc_summary_pub.txt"
LCDB_CACHE    = "/content/asteroid-pipeline/data/lcdb_cache.csv"


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class LCDBRecord:
    """
    One asteroid record from the LCDB.

    Attributes
    ----------
    provid      : designation used for lookup
    number      : MPC number (None if unnumbered)
    name        : name (None if unnamed)
    period_hr   : rotation period in hours (NaN if unknown)
    u_code      : integer reliability code (0=not in LCDB, 1-3)
    u_flag      : raw U string e.g. "3", "2+", "3-"
    amp_min     : minimum lightcurve amplitude in mag (NaN if unknown)
    amp_max     : maximum lightcurve amplitude in mag (NaN if unknown)
    taxonomy    : taxonomic class e.g. "S", "C" (None if unknown)
    hg_slope_G  : HG phase slope G (NaN if unknown)
    is_binary   : True if known or suspected binary
    found       : True if record was found in LCDB
    """
    provid:     str
    number:     Optional[int]
    name:       Optional[str]
    period_hr:  float
    u_code:     int
    u_flag:     str
    amp_min:    float
    amp_max:    float
    taxonomy:   Optional[str]
    hg_slope_G: float
    is_binary:  bool
    found:      bool


# ── Download and load ─────────────────────────────────────────────────────────

def download_lcdb(cache_path: str = LCDB_CACHE) -> str:
    """Download LCDB and cache lc_summary_pub.txt locally."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        logger.info(f"LCDB cache found at {cache_path} — skipping download")
        return cache_path

    logger.info("Downloading LCDB...")
    response = requests.get(LCDB_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open(LCDB_FILENAME) as f:
            content = f.read().decode("latin-1")

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"LCDB cached at {cache_path}")
    return cache_path


def load_lcdb(cache_path: str = LCDB_CACHE) -> pd.DataFrame:
    """
    Load cached LCDB into a clean DataFrame.

    Returns DataFrame with columns:
    NUMBER, NAME, DESIG, CLASS, H, G, PERIOD, AMIN, AMAX, U, BIN
    """
    if not os.path.exists(cache_path):
        download_lcdb(cache_path)

    # Fixed-width file:
    # Line 0: title, Line 1: date, Line 2: blank, Line 3: headers
    # Line 4: dashes, Line 5+: data
    df = pd.read_fwf(
        cache_path,
        skiprows=[0, 1, 2, 4],
        header=0,
        encoding="utf-8",
        dtype=str,
    )
    df.columns = df.columns.str.strip()

    logger.info(f"LCDB loaded: {len(df):,} records")
    return df


# ── MPC designation utilities ─────────────────────────────────────────────────

def _unpack_mpc_desig(packed: str) -> str:
    """
    Convert packed MPC provisional designation to unpacked form.

    Examples:
      "J95X00A" → "1995 XA"      (standard packed)
      "K04A00A" → "2004 AA"
      "A910"    → old format, return as-is for now

    This handles the most common cases. Unusual formats
    (survey designations, comets) are returned unchanged.
    """
    if not packed or pd.isna(packed):
        return ""

    packed = str(packed).strip()

    # Already unpacked (contains space)
    if " " in packed:
        return packed

    # Must be at least 5 chars for standard packed format
    if len(packed) < 5:
        return packed

    # Standard packed: letter + 2-digit year + letters/digits
    # e.g. K04A00A = 2004 AA, J95X00A = 1995 XA
    century_map = {
        "I": "18", "J": "19", "K": "20"
    }

    if packed[0] in century_map and packed[1:3].isdigit():
        century = century_map[packed[0]]
        year    = century + packed[1:3]

        if len(packed) >= 7:
            half   = packed[3]        # A-H, J-Y
            sub    = packed[6]        # second letter
            order  = packed[4:6]      # order digits

            order_int = 0
            try:
                order_int = int(order)
            except ValueError:
                pass

            if order_int == 0:
                order_str = ""
            else:
                order_str = str(order_int)

            unpacked = f"{year} {half}{sub}{order_str}"
            return unpacked.strip()

    return packed


def _normalise(s: str) -> str:
    """Normalise designation for comparison: uppercase, single spaces."""
    if not s or pd.isna(s):
        return ""
    return " ".join(str(s).upper().strip().split())


# ── Lookup ────────────────────────────────────────────────────────────────────

def lookup(provid: str, df_lcdb: pd.DataFrame) -> LCDBRecord:
    """
    Look up one asteroid in the LCDB by provisional designation or number.

    Tries:
    1. Direct match on DESIG column (packed format)
    2. Unpacked DESIG match
    3. NAME column match
    4. NUMBER column match (if provid is numeric)

    Parameters
    ----------
    provid   : provisional designation e.g. "2025 MA19" or MPC number
    df_lcdb  : DataFrame from load_lcdb()

    Returns
    -------
    LCDBRecord with found=False if not in LCDB
    """
    search_norm = _normalise(provid)

    # Build normalised columns for matching (compute once per lookup — fast enough)
    desig_norm  = df_lcdb["DESIG"].apply(_normalise)
    desig_unp   = df_lcdb["DESIG"].apply(
        lambda x: _normalise(_unpack_mpc_desig(str(x))) if pd.notna(x) else ""
    )
    name_norm   = df_lcdb["NAME"].apply(_normalise)
    number_norm = df_lcdb["NUMBER"].apply(_normalise)

    # Try each matching strategy
    row = None
    for col_series in [desig_norm, desig_unp, name_norm, number_norm]:
        mask = col_series == search_norm
        if mask.any():
            row = df_lcdb[mask].iloc[0]
            break

    if row is None:
        logger.debug(f"{provid}: not found in LCDB")
        return LCDBRecord(
            provid=provid, number=None, name=None,
            period_hr=np.nan, u_code=0, u_flag="",
            amp_min=np.nan, amp_max=np.nan,
            taxonomy=None, hg_slope_G=np.nan,
            is_binary=False, found=False,
        )

    # ── Parse fields ──────────────────────────────────────────────────────────

    def _float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    def _int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _str(val):
        s = str(val).strip() if pd.notna(val) else ""
        return s if s and s.lower() != "nan" else ""

    period_hr  = _float(row.get("PERIOD"))
    amp_min    = _float(row.get("AMIN"))
    amp_max    = _float(row.get("AMAX"))
    number     = _int(row.get("NUMBER"))
    name       = _str(row.get("NAME")) or None
    taxonomy   = _str(row.get("CLASS")) or None
    g_val      = _float(row.get("G"))

    # U code
    u_raw  = _str(row.get("U"))
    u_code = 0
    u_flag = u_raw
    if u_raw:
        try:
            base = int(u_raw[0])
            # 3- means downgrade: the period may be wrong
            u_code = max(base - 1, 1) if u_raw.endswith("-") else base
        except (ValueError, IndexError):
            pass

    # Binary flag — BIN column contains B (binary) or M (moonlet/satellite)
    bin_val   = _str(row.get("BIN"))
    is_binary = bin_val in ("B", "M", "Y")

    # G slope from taxonomy if not directly available
    hg_slope_G = g_val
    if np.isnan(hg_slope_G) and taxonomy:
        t = taxonomy.upper()
        if any(c in t for c in ["C", "B", "F", "G"]):
            hg_slope_G = 0.10
        elif any(c in t for c in ["S", "Q", "R", "V"]):
            hg_slope_G = 0.15
        elif any(c in t for c in ["E", "M"]):
            hg_slope_G = 0.25

    logger.debug(
        f"{provid}: found in LCDB — "
        f"P={period_hr:.3f}hr U={u_flag} amp=[{amp_min},{amp_max}] "
        f"tax={taxonomy} binary={is_binary}"
    )

    return LCDBRecord(
        provid=provid, number=number, name=name,
        period_hr=period_hr, u_code=u_code, u_flag=u_flag,
        amp_min=amp_min, amp_max=amp_max,
        taxonomy=taxonomy, hg_slope_G=hg_slope_G,
        is_binary=is_binary, found=True,
    )


def lookup_batch(provids: list, df_lcdb: pd.DataFrame) -> dict:
    """Look up multiple asteroids. Returns {provid: LCDBRecord}."""
    return {p: lookup(p, df_lcdb) for p in provids}


# ── Pipeline comparison ───────────────────────────────────────────────────────

def compare_to_lcdb(
    pipeline_period: float,
    lcdb_record:     LCDBRecord,
    tolerance:       float = 0.05,
) -> dict:
    """
    Compare a pipeline period to the LCDB known period.

    Checks direct agreement, P/2 alias (pipeline found half-period),
    and 2P alias (pipeline found double period).

    Returns
    -------
    dict with keys:
      agreement    : "exact" | "half_period" | "double_period" |
                     "disagree" | "no_prior"
      delta_pct    : fractional difference from LCDB period
      within_tol   : True if within tolerance
      is_half      : True if pipeline found P/2
      is_double    : True if pipeline found 2P
      note         : human-readable summary
    """
    if not lcdb_record.found or np.isnan(lcdb_record.period_hr):
        return dict(
            agreement="no_prior", delta_pct=np.nan,
            within_tol=None, is_half=False, is_double=False,
            note="No LCDB record available",
        )

    P_known = lcdb_record.period_hr
    P_pipe  = pipeline_period

    delta_pct    = abs(P_pipe - P_known)          / P_known
    delta_half   = abs(P_pipe - P_known / 2.0)    / (P_known / 2.0)
    delta_double = abs(P_pipe - P_known * 2.0)    / (P_known * 2.0)

    within_tol = delta_pct    <= tolerance
    is_half    = delta_half   <= tolerance
    is_double  = delta_double <= tolerance

    if within_tol:
        agreement = "exact"
        note = (f"Pipeline {P_pipe:.3f}hr agrees with LCDB "
                f"{P_known:.3f}hr (U={lcdb_record.u_flag})")
    elif is_half:
        agreement = "half_period"
        note = (f"Pipeline {P_pipe:.3f}hr = P/2 of LCDB "
                f"{P_known:.3f}hr — double-hump alias")
    elif is_double:
        agreement = "double_period"
        note = (f"Pipeline {P_pipe:.3f}hr = 2P of LCDB "
                f"{P_known:.3f}hr")
    else:
        agreement = "disagree"
        note = (f"Pipeline {P_pipe:.3f}hr disagrees with LCDB "
                f"{P_known:.3f}hr (Δ={delta_pct*100:.1f}%)")

    return dict(
        agreement=agreement, delta_pct=delta_pct,
        within_tol=within_tol, is_half=is_half,
        is_double=is_double, note=note,
    )
