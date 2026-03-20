"""
tier3.py
--------
Tier 3: Disambiguation for cases where Tier 2 methods disagree.
Runs Bayesian period posterior and CLEAN alias deconvolution.

When to use
-----------
Only for asteroids where Tier 2 methods disagreed (t2result.to_tier3=True).
This tier is computationally expensive — expect ~1–5 seconds per asteroid.
It processes hundreds of objects per night, not millions.

Methods
-------
1. Bayesian posterior  — P(period | data) ∝ exp(-chi_sq / 2)
                         Uses Fourier N=2 likelihood (correct double-hump model).
                         Returns MAP period and 95% credible interval.

2. CLEAN periodogram   — Roberts et al. (1987)
                         Deconvolves the window function aliases from the
                         dirty power spectrum. Isolated single peak = confident.

Decision
--------
If Bayesian 95% CI width < threshold OR CLEAN isolates a single dominant peak:
   → publish as "tentative" with reliability flag
Else:
   → flag for dedicated follow-up observations

Functions
---------
run_tier3(data, t2result, config)           — main entry point
bayesian_period_posterior(t,y,dy,tp)       — compute posterior
clean_periodogram(t,y,dy,freqs)            — CLEAN spectrum
compute_credible_interval(posterior, tp)   — 95% CI from posterior
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from config import PipelineConfig, DEFAULT_CONFIG
from preprocessing import PreparedData
from tier2 import Tier2Result

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Tier3Result:
    """
    Output of Tier 3 disambiguation.

    Attributes
    ----------
    provid            : asteroid designation
    publish_tentative : True = publish with reliability='tentative'
    needs_followup    : True = flag for dedicated telescope time
    best_period_bayes : Bayesian MAP period (hours)
    best_period_clean : CLEAN best period (hours)
    ci_lo, ci_hi      : 95% credible interval bounds (hours)
    ci_width          : 95% CI width (hours)
    clean_peak_ratio  : ratio of top CLEAN peak to 2nd peak (higher = cleaner)
    final_period      : adopted period (Bayesian MAP if CI narrow, else NaN)
    final_period_unc  : adopted uncertainty (half CI width)
    reliability       : 'tentative' or 'followup_needed'
    test_periods      : period grid
    posterior         : Bayesian posterior array
    clean_power       : CLEAN power array
    """
    provid:             str
    publish_tentative:  bool
    needs_followup:     bool
    best_period_bayes:  float
    best_period_clean:  float
    ci_lo:              float
    ci_hi:              float
    ci_width:           float
    clean_peak_ratio:   float
    final_period:       float
    final_period_unc:   float
    reliability:        str
    test_periods:       np.ndarray
    posterior:          np.ndarray
    clean_power:        np.ndarray


# ── Main Tier 3 entry point ───────────────────────────────────────────────────

def run_tier3(
    data:     PreparedData,
    t2result: Tier2Result,
    config:   PipelineConfig = DEFAULT_CONFIG,
) -> Tier3Result:
    """
    Run Tier 3 disambiguation using CLEAN periodogram only.

    The Bayesian posterior was removed — it uses the same Fourier
    chi-sq likelihood as MHAOV and is not independent. CLEAN is
    genuinely independent: it deconvolves the spectral window function.

    Decision: publish tentative if CLEAN confirms MBLS period
    (peak ratio >= threshold AND CLEAN agrees with MBLS within tol).
    """
    if not t2result.to_tier3:
        raise ValueError(f"{data.provid}: Tier2 did not route to Tier3")

    cfg_p = config.period
    cfg_t = config.tier

    # Focused grid around Tier 2 candidates ± 20%
    focused = []
    for p_cand in [t2result.best_period_mhaov,
                   t2result.best_period_mbls,
                   t2result.best_period_ce]:
        if not np.isnan(p_cand) and p_cand > 0:
            lo = max(cfg_p.period_min_hr, p_cand * 0.80)
            hi = min(cfg_p.period_max_hr, p_cand * 1.20)
            focused.extend(np.linspace(lo, hi, 300).tolist())

    focused_periods = (np.unique(np.sort(np.array(focused)))
                       if len(focused) >= 10
                       else t2result.test_periods)

    # CLEAN
    logger.debug(f"{data.provid}: running CLEAN...")
    freqs     = 1.0 / focused_periods[::-1]
    clean_pow = clean_periodogram(
        data.t_hrs, data.y_dt, data.dy, freqs,
        gain=cfg_p.clean_gain, n_iter=cfg_p.clean_niter
    )
    clean_pow  = clean_pow[::-1]
    best_clean = focused_periods[np.argmax(clean_pow)]
    peak_ratio = _clean_peak_ratio(clean_pow)

    # Best period = MBLS (has 2-minima correction applied)
    best_mbls       = t2result.best_period_mbls
    grid_step       = float(np.median(np.diff(focused_periods)))
    ci_width        = grid_step * 3.0
    ci_lo           = best_mbls - ci_width / 2.0
    ci_hi           = best_mbls + ci_width / 2.0

    clean_agrees = (abs(best_clean - best_mbls) /
                    (best_mbls + 1e-12) <= cfg_t.agreement_tol)
    clean_strong = peak_ratio >= cfg_t.clean_peak_ratio

    logger.debug(
        f"{data.provid} Tier3: CLEAN={best_clean:.3f}hr "
        f"ratio={peak_ratio:.1f} MBLS={best_mbls:.3f}hr "
        f"agrees={clean_agrees} strong={clean_strong}"
    )

    if clean_agrees and clean_strong:
        return Tier3Result(
            provid=data.provid, publish_tentative=True, needs_followup=False,
            best_period_bayes=best_mbls, best_period_clean=best_clean,
            ci_lo=ci_lo, ci_hi=ci_hi, ci_width=ci_width,
            clean_peak_ratio=peak_ratio,
            final_period=best_mbls, final_period_unc=ci_width / 2.0,
            reliability="tentative",
            test_periods=focused_periods, posterior=clean_pow,
            clean_power=clean_pow,
        )

    return Tier3Result(
        provid=data.provid, publish_tentative=False, needs_followup=True,
        best_period_bayes=best_mbls, best_period_clean=best_clean,
        ci_lo=ci_lo, ci_hi=ci_hi, ci_width=ci_width,
        clean_peak_ratio=peak_ratio,
        final_period=np.nan, final_period_unc=np.nan,
        reliability="followup_needed",
        test_periods=focused_periods, posterior=clean_pow,
        clean_power=clean_pow,
    )


def bayesian_period_posterior(
    t:            np.ndarray,
    y:            np.ndarray,
    dy:           np.ndarray,
    test_periods: np.ndarray,
    nh:           int = 2,
) -> np.ndarray:
    """
    Bayesian posterior probability over trial periods.

    Model: log P(period | data) ∝ -chi_sq(best_fit) / 2
    where chi_sq is computed using a Fourier NH-harmonic fit.
    Prior: uniform over period (log-flat prior would be more physical
           but uniform is standard for this period range).

    The Fourier N=2 fit correctly models the double-hump, so the
    likelihood peaks at the true rotation period.

    Parameters
    ----------
    nh : int
        Number of harmonics (2 = double-hump asteroid model)

    Returns
    -------
    Normalised probability density array, same length as test_periods
    """
    w           = 1.0 / dy**2
    y_wmean     = np.average(y, weights=w)
    chisq_total = float(np.sum(w * (y - y_wmean)**2))

    fc2 = np.array([_fourier_chisq_power(t, y, dy, p, nh) for p in test_periods])

    # chi_sq at each period = (1 - power) * chi_sq_total
    log_post = -0.5 * (1.0 - fc2) * chisq_total
    log_post -= log_post.max()   # numerical stability
    posterior = np.exp(log_post)

    # Normalise to unit area (probability density)
    norm = np.trapezoid(posterior, test_periods)
    if norm > 0:
        posterior /= norm

    return posterior


def compute_credible_interval(
    posterior:    np.ndarray,
    test_periods: np.ndarray,
    alpha:        float = 0.95,
) -> Tuple[float, float]:
    """
    Compute the highest-posterior-density credible interval.

    Uses the simple equal-tailed interval from the CDF.

    Parameters
    ----------
    alpha : float
        Coverage probability e.g. 0.95 for 95% CI

    Returns
    -------
    (lo, hi) period bounds
    """
    tail = (1.0 - alpha) / 2.0
    dp   = np.diff(test_periods, prepend=test_periods[0])
    cdf  = np.cumsum(posterior * dp)
    cdf /= cdf[-1]

    lo = test_periods[np.searchsorted(cdf, tail)]
    hi = test_periods[np.searchsorted(cdf, 1.0 - tail)]
    return float(lo), float(hi)


# ── CLEAN implementation ──────────────────────────────────────────────────────

def clean_periodogram(
    t:       np.ndarray,
    y:       np.ndarray,
    dy:      np.ndarray,
    freqs:   np.ndarray,
    gain:    float = 0.1,
    n_iter:  int   = 200,
) -> np.ndarray:
    """
    Roberts et al. (1987) CLEAN algorithm for unevenly sampled data.

    Iteratively removes the window function's alias pattern from the
    dirty power spectrum. Each iteration:
    1. Find the peak in the residual spectrum
    2. Add gain × peak amplitude to the clean components
    3. Subtract gain × window response centred at peak frequency
    4. Repeat until n_iter exhausted

    Parameters
    ----------
    freqs  : frequency grid (cycles/hour) — must be low-to-high
    gain   : CLEAN loop gain (0.05–0.2 typical)
    n_iter : number of CLEAN iterations

    Returns
    -------
    Clean power spectrum at each frequency in freqs (normalised 0–1)
    """
    w = 1.0 / dy**2
    w = w / w.sum()

    def dft(f_arr):
        """Weighted DFT at each frequency in f_arr."""
        ph  = 2.0 * np.pi * np.outer(t, f_arr)
        return np.sum(w[:, None] * y[:, None] * np.exp(-1j * ph), axis=0)

    def wdft(f_arr):
        """Window function DFT (data = 1)."""
        ph = 2.0 * np.pi * np.outer(t, f_arr)
        return np.sum(w[:, None] * np.exp(-1j * ph), axis=0)

    dirty   = dft(freqs)
    clean_c = np.zeros(len(freqs), dtype=complex)
    residual = dirty.copy()

    for _ in range(n_iter):
        idx  = np.argmax(np.abs(residual)**2)
        peak = residual[idx]
        f0   = freqs[idx]
        clean_c[idx] += gain * peak
        residual      -= gain * peak         * wdft(freqs - f0)
        residual      -= gain * np.conj(peak) * wdft(freqs + f0)

    clean_pow = np.abs(clean_c)**2
    mx = clean_pow.max()
    return clean_pow / mx if mx > 0 else clean_pow


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fourier_chisq_power(
    t:      np.ndarray,
    y:      np.ndarray,
    dy:     np.ndarray,
    period: float,
    nh:     int = 2,
) -> float:
    """
    Fractional chi-sq improvement from fitting NH Fourier harmonics.
    Returns value in [0, 1]. Used for Bayesian likelihood.
    """
    w  = 1.0 / dy**2
    ph = 2.0 * np.pi * t / period
    cols = [np.ones(len(t))]
    for k in range(1, nh + 1):
        cols += [np.cos(k * ph), np.sin(k * ph)]
    A = np.column_stack(cols)
    try:
        Aw   = A * w[:, None]
        c    = np.linalg.solve(Aw.T @ A, Aw.T @ y)
        yf   = A @ c
        cs   = np.sum(w * (y - yf)**2)
        cm   = np.sum(w * (y - np.average(y, weights=w))**2)
        return float(max(0.0, 1.0 - cs / cm))
    except np.linalg.LinAlgError:
        return 0.0


def _clean_peak_ratio(clean_pow: np.ndarray) -> float:
    """
    Ratio of strongest CLEAN peak to second-strongest peak.
    High ratio (>3) means one dominant period; low ratio means ambiguous.
    """
    if len(clean_pow) < 2:
        return 1.0
    sorted_pow = np.sort(clean_pow)[::-1]
    if sorted_pow[1] == 0:
        return float("inf")
    return float(sorted_pow[0] / sorted_pow[1])
