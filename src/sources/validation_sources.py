"""
validation_sources.py
---------------------
Additional period validation sources beyond LCDB.

Status: STUB — not yet implemented (Change 7 roadmap).

Sources
-------
1. DAMIT — Database of Asteroid Models from Inversion Techniques
   URL: https://astro.troja.mff.cuni.cz/projects/damit/
   ~3,000 objects with spin-state solutions (period + pole + shape).
   Periods are high-confidence (equivalent to LCDB U=3).
   API: https://astro.troja.mff.cuni.cz/projects/damit/api/

2. JPL Small-Body Database
   URL: https://ssd.jpl.nasa.gov/tools/sbdb_query.cgi
   rot_per field sourced from multiple surveys.
   REST API: https://ssd-api.jpl.nasa.gov/sbdb_query.api
   Example: https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=pdes,rot_per,rot_per_sig

Usage in validation
-------------------
These sources supplement LCDB for validating pipeline results on objects
where LCDB has no record or low-confidence (U=1) periods.

Priority order for ground truth:
  1. DAMIT (spin inversion — most rigorous)
  2. LCDB U=3
  3. JPL SBDB rot_per (aggregated, source not always traceable)
  4. LCDB U=2
  5. Greenstreet et al. 2026 (for RFL objects specifically)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PeriodRecord:
    """
    Standardised period record from any validation source.
    Allows uniform comparison regardless of source.
    """
    provid:     str
    period_hr:  float
    period_unc: float        # uncertainty in hours (NaN if unknown)
    source:     str          # "DAMIT", "JPL_SBDB", "LCDB", "Greenstreet2026"
    confidence: str          # "high", "medium", "low"
    reference:  str          # citation or URL
    found:      bool


def lookup_damit(provid: str) -> PeriodRecord:
    """
    Look up rotation period from DAMIT.

    NOT YET IMPLEMENTED — placeholder for Change 7.

    DAMIT contains spin-state solutions from lightcurve inversion.
    These are the most rigorously determined periods available —
    they fit not just the period but the full 3D shape and pole direction.
    """
    raise NotImplementedError(
        "DAMIT lookup not yet implemented. See Change 7 in README roadmap.\n"
        "API: https://astro.troja.mff.cuni.cz/projects/damit/api/"
    )


def lookup_jpl_sbdb(provid: str) -> PeriodRecord:
    """
    Look up rotation period from JPL Small-Body Database.

    NOT YET IMPLEMENTED — placeholder for Change 7.

    JPL SBDB aggregates period data from multiple sources. The rot_per
    field includes a sigma estimate when available. Note that the original
    source is not always traceable — treat as medium confidence unless
    cross-checked against primary literature.
    """
    raise NotImplementedError(
        "JPL SBDB lookup not yet implemented. See Change 7 in README roadmap.\n"
        "API: https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    )


def lookup_all_sources(
    provid: str,
    df_lcdb=None,
) -> list:
    """
    Look up period from all available validation sources.

    Returns a list of PeriodRecord objects sorted by confidence.
    The caller should use the highest-confidence record for validation.

    NOT YET IMPLEMENTED for DAMIT and JPL SBDB — placeholder for Change 7.
    LCDB lookup already implemented in sources/lcdb.py.
    """
    records = []

    # LCDB (already implemented)
    if df_lcdb is not None:
        try:
            from sources.lcdb import lookup as lcdb_lookup
            rec = lcdb_lookup(provid, df_lcdb)
            if rec.found:
                confidence = "high" if rec.u_code >= 3 else (
                    "medium" if rec.u_code == 2 else "low"
                )
                records.append(PeriodRecord(
                    provid=provid,
                    period_hr=rec.period_hr,
                    period_unc=np.nan,
                    source="LCDB",
                    confidence=confidence,
                    reference="Warner et al., LCDB",
                    found=True,
                ))
        except Exception as e:
            logger.debug(f"{provid}: LCDB lookup failed: {e}")

    # DAMIT — not yet implemented
    # JPL SBDB — not yet implemented

    return records
