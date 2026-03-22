"""
ztf.py
------
Fetch ZTF photometry for asteroid period analysis.

Status: STUB — not yet implemented (Change 7 roadmap).

Data source
-----------
ZTF public photometry via IRSA lightcurve API:
  https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html

Query by MPC designation or sky coordinates. Returns g/r/i photometry
with ~mag 20.5 depth. Baseline since 2018 — 7+ year arc available for
objects observed throughout the survey.

Key design considerations
-------------------------
1. Zero-point offsets: ZTF g/r/i and Rubin Lg/Lr/Li are close but not
   identical (~0.02–0.05 mag offsets). MBLS absorbs small per-band
   offsets internally, but large systematics should be corrected first.

2. Source tagging: all ZTF rows must carry source="ZTF" so characterise.py
   can count n_sources and unlock regime="combined" in _classify_regime().

3. Epoch correction: heliocentric distance changes between Rubin and ZTF
   epochs (potentially years apart) must be corrected before combining.
   Use geometry.apply_reduced_magnitude() after fetching r/delta from
   JPL Horizons for each epoch.

4. Methodological principle: ZTF augmentation must be triggered by data
   quality criteria (n_obs < threshold, regime == "sparse", baseline < X)
   BEFORE running the period search — not after seeing a failed result.
   The decision to fetch must be independent of the period outcome.

Planned API
-----------
fetch_ztf(provid, config)     — fetch ZTF lightcurve for one asteroid
merge_with_rubin(df_rubin, df_ztf, config) — combine and tag by source
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ZTF IRSA lightcurve API base URL
ZTF_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

# ZTF band name mapping to canonical pipeline names
ZTF_BAND_MAP = {
    "zg": "Lg",   # ZTF g → Rubin Lg equivalent
    "zr": "Lr",   # ZTF r → Rubin Lr equivalent
    "zi": "Li",   # ZTF i → Rubin Li equivalent
}

# Approximate ZTF–Rubin zero-point offsets (mag) from cross-matching
# These are approximate — proper cross-calibration should use contemporaneous
# observations of field stars. Values from Ofek et al. (2012) and
# Bellm et al. (2019).
ZTF_RUBIN_OFFSETS = {
    "Lg": -0.02,   # ZTF g is ~0.02 mag brighter than Rubin g on average
    "Lr": -0.01,
    "Li":  0.03,
}


def fetch_ztf(
    provid: str,
    config=None,
) -> pd.DataFrame:
    """
    Fetch ZTF photometry for a single asteroid from IRSA.

    NOT YET IMPLEMENTED — placeholder for Change 7.

    Parameters
    ----------
    provid : str
        MPC provisional designation e.g. "2025 MA19"
    config : PipelineConfig, optional

    Returns
    -------
    pd.DataFrame with columns: provid, mjd, band, mag, rmsmag, source
    Empty DataFrame if object not found or API unavailable.
    """
    raise NotImplementedError(
        "ZTF fetcher not yet implemented. See Change 7 in README roadmap.\n"
        "Planned: query IRSA API at https://irsa.ipac.caltech.edu/"
        "docs/program_interface/ztf_lightcurve_api.html"
    )


def merge_with_rubin(
    df_rubin: pd.DataFrame,
    df_ztf: pd.DataFrame,
    apply_offsets: bool = True,
) -> pd.DataFrame:
    """
    Combine Rubin and ZTF observations into a single tagged DataFrame.

    NOT YET IMPLEMENTED — placeholder for Change 7.

    Tags Rubin rows as source="Rubin" and ZTF rows as source="ZTF".
    When n_sources > 1, characterise() will classify regime as "combined"
    which enables the full pipeline with highest reliability ceiling.

    Parameters
    ----------
    df_rubin : pd.DataFrame
        Rubin observations from ingestion
    df_ztf : pd.DataFrame
        ZTF observations from fetch_ztf()
    apply_offsets : bool
        If True, apply ZTF_RUBIN_OFFSETS to ZTF magnitudes before merging

    Returns
    -------
    pd.DataFrame with source column added, sorted by mjd
    """
    raise NotImplementedError(
        "merge_with_rubin not yet implemented. See Change 7 in README roadmap."
    )
