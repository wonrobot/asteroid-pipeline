"""
precompute.py
-------------
Saves all plot-ready arrays for one asteroid during the pipeline run.
Called from run_single_asteroid when output_plots_dir is set.

Output files per asteroid (all in output_plots_dir):
  obs_{safe_provid}.npz    — raw observations + preprocessed arrays
  pgram_{safe_provid}.npz  — all periodograms + window function + CLEAN
  fold_{safe_provid}.npz   — phase-folded lightcurve at adopted period
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _safe_name(provid: str) -> str:
    """Convert provid to safe filename: '2025 MA19' → '2025_MA19'."""
    return provid.replace(' ', '_').replace('/', '_')


def save_obs(provid, data, output_dir):
    """Save raw + preprocessed observation arrays."""
    fname = os.path.join(output_dir, f"obs_{_safe_name(provid)}.npz")
    np.savez_compressed(
        fname,
        provid       = np.array([provid]),
        mjd          = data.df["mjd"].values,
        t_hrs        = data.t_hrs,
        mag_raw      = data.df["mag"].values,
        dy           = data.dy,
        bands        = np.array(data.bands, dtype=str),
        y_dt         = data.y_dt,
        y_multiband  = data.y_multiband,
        poly_coeffs  = data.poly_coeffs,
        baseline_hr  = np.array([data.baseline_hr]),
        n_obs        = np.array([data.n_obs]),
        geometry_applied = np.array([data.geometry_applied]),
    )
    logger.debug(f"{provid}: saved obs arrays → {fname}")


def save_periodograms(provid, t1result, t2result, t3result, data, output_dir):
    """Save all periodogram arrays + window function + CLEAN if ran."""
    from window import compute_window_function

    fname = os.path.join(output_dir, f"pgram_{_safe_name(provid)}.npz")

    # Window function on T2 grid (or T1 if T2 didn't run)
    ref_periods = (t2result.test_periods
                   if t2result is not None and len(t2result.test_periods) > 0
                   else t1result.test_periods)
    window_power = np.array([])
    if len(ref_periods) > 0:
        try:
            window_power = compute_window_function(data.t_hrs, ref_periods)
        except Exception as e:
            logger.warning(f"{provid}: window function failed ({e})")

    arrays = dict(
        provid = np.array([provid]),
        # Tier 1
        t1_periods    = t1result.test_periods,
        t1_gls_power  = t1result.gls_power,
        t1_mbls_power = t1result.mbls_power,
        # Window (same grid as T2/T1)
        window_periods = ref_periods,
        window_power   = window_power,
    )

    if t2result is not None:
        arrays.update(dict(
            t2_periods    = t2result.test_periods,
            t2_mhaov_power = t2result.mhaov_power,
            t2_mbls_power  = t2result.mbls_power,
            t2_ce_scores   = t2result.ce_scores,
            t2_ce_periods  = np.linspace(
                t2result.test_periods[0] if len(t2result.test_periods)>0 else 0.5,
                t2result.test_periods[-1] if len(t2result.test_periods)>0 else 24.0,
                len(t2result.ce_scores) if len(t2result.ce_scores)>0 else 0
            ),
        ))

    if t3result is not None:
        arrays.update(dict(
            t3_periods    = t3result.test_periods,
            t3_clean_power = t3result.clean_power,
        ))

    np.savez_compressed(fname, **arrays)
    logger.debug(f"{provid}: saved periodogram arrays → {fname}")


def save_phase_fold(provid, data, final_period_hr, t2result, output_dir):
    """Save phase-folded lightcurve at adopted period."""
    if np.isnan(final_period_hr) or final_period_hr <= 0:
        return

    fname = os.path.join(output_dir, f"fold_{_safe_name(provid)}.npz")

    phase = (data.t_hrs % final_period_hr) / final_period_hr
    sort_idx = np.argsort(phase)

    # Fitted curve from MBLS if available
    fitted_phase  = np.array([])
    fitted_mag    = np.array([])
    if t2result is not None and not np.isnan(t2result.best_period_mbls):
        try:
            from gatspy.periodic import LombScargleMultiband
            from config import DEFAULT_CONFIG
            model = LombScargleMultiband(Nterms_base=2, Nterms_band=0)
            model.fit(data.t_hrs, data.y_multiband, data.dy, data.bands)
            fitted_phase = np.linspace(0, 1, 500)
            t_grid = fitted_phase * final_period_hr
            import numpy as np2
            unique_bands, counts = np.unique(data.bands, return_counts=True)
            dominant = unique_bands[np.argmax(counts)]
            fitted_mag = model.predict(
                t_grid,
                filts=np.full(500, dominant),
                period=final_period_hr,
            )
        except Exception as e:
            logger.warning(f"{provid}: MBLS phase fold model failed ({e})")

    np.savez_compressed(
        fname,
        provid        = np.array([provid]),
        period_hr     = np.array([final_period_hr]),
        phase         = phase[sort_idx],
        y_dt          = data.y_dt[sort_idx],
        y_multiband   = data.y_multiband[sort_idx],
        dy            = data.dy[sort_idx],
        bands         = np.array(data.bands[sort_idx], dtype=str),
        fitted_phase  = fitted_phase,
        fitted_mag    = fitted_mag,
    )
    logger.debug(f"{provid}: saved phase fold → {fname}")


def save_all(provid, data, t1result, t2result, t3result, rel, output_dir):
    """Save all pre-computed arrays for one asteroid."""
    os.makedirs(output_dir, exist_ok=True)
    save_obs(provid, data, output_dir)
    save_periodograms(provid, t1result, t2result, t3result, data, output_dir)
    final_p = rel.period_hr if rel is not None else np.nan
    save_phase_fold(provid, data, final_p, t2result, output_dir)
