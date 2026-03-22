"""
atlas.py
--------
Fetch ATLAS photometry for asteroid period analysis.

Status: STUB — not yet implemented (Change 7 roadmap).

Data source
-----------
ATLAS forced photometry API (requires free registration):
  https://fallingstar-data.com/forcedphot/

ATLAS (Asteroid Terrestrial-impact Last Alert System) observes the full
sky roughly every 2 days in o (orange, 560–820nm) and c (cyan, 420–650nm)
broadband filters. Depth ~mag 19 per exposure. Running since 2015.

Key design considerations
-------------------------
1. Non-standard bands: ATLAS o/c bands do not map directly to Rubin gri.
   They are broad filters spanning multiple standard bands. Direct merging
   into the gri multiband fit is not appropriate.
   
   Recommended approach: treat ATLAS as an INDEPENDENT periodogram check,
   not as additional data points in the joint MBLS fit. If ATLAS independently
   confirms the Rubin period, reliability increases. If it disagrees, flag
   for investigation.

2. API access: registration required at https://fallingstar-data.com/forcedphot/
   The API returns forced photometry at a specified sky position — suitable
   for asteroids only when ephemeris positions are available (JPL Horizons).

3. Best use cases:
   - Bright asteroids (H < 18, mag < 19) where Rubin may saturate
   - Long-period objects (P > 10hr) needing multi-year baseline
   - Cross-check for objects where Rubin gives ambiguous results

Planned API
-----------
fetch_atlas(provid, config)   — fetch ATLAS lightcurve for one asteroid
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ATLAS_API_URL = "https://fallingstar-data.com/forcedphot/queue/"

ATLAS_BAND_MAP = {
    "o": "ATLASo",   # orange — keep separate from Rubin bands
    "c": "ATLASc",   # cyan   — keep separate from Rubin bands
}


def fetch_atlas(
    provid: str,
    config=None,
) -> pd.DataFrame:
    """
    Fetch ATLAS forced photometry for a single asteroid.

    NOT YET IMPLEMENTED — placeholder for Change 7.

    Requires:
    - ATLAS API credentials (free registration)
    - JPL Horizons ephemeris to get sky positions at each epoch

    Parameters
    ----------
    provid : str
        MPC provisional designation
    config : PipelineConfig, optional

    Returns
    -------
    pd.DataFrame with columns: provid, mjd, band, mag, rmsmag, source
    Bands will be "ATLASo" and "ATLASc" — NOT merged with Rubin gri.
    """
    raise NotImplementedError(
        "ATLAS fetcher not yet implemented. See Change 7 in README roadmap.\n"
        "Planned: query https://fallingstar-data.com/forcedphot/ after "
        "obtaining sky positions from JPL Horizons for each epoch."
    )
