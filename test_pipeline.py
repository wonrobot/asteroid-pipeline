"""
tests/test_pipeline.py
----------------------
Unit tests for each pipeline module.
Run with: pytest tests/test_pipeline.py -v

Tests cover
-----------
- GLS power at known period
- MHAOV F-statistic at known period
- Conditional entropy at known period
- Preprocessing: band offsets, detrend
- Tier1: passes / rejects correctly
- Tier2: agreement check
- Catalog: row creation, upsert
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pandas as pd
import pytest

from config import PipelineConfig, DEFAULT_CONFIG
from tier1 import gls_power, gls_periodogram
from tier2 import mhaov_single, ce_single, check_agreement
from tier3 import bayesian_period_posterior, compute_credible_interval
from preprocessing import preprocess, compute_snr, _apply_band_offsets, _geometry_detrend
from catalog import result_to_row, append_result, init_catalog


# ── Synthetic data generator ──────────────────────────────────────────────────

def make_synthetic_lc(
    period_hr: float = 4.8,
    amplitude: float = 0.3,
    n_obs:     int   = 100,
    noise:     float = 0.03,
    seed:      int   = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic asteroid lightcurve with known period.
    Uses a double-hump shape (2nd harmonic dominant).
    """
    rng = np.random.default_rng(seed)
    # Sparse nightly observations over 10 nights
    nights = np.sort(rng.choice(range(10), size=n_obs, replace=True))
    t_days = nights + rng.uniform(0, 0.1, n_obs)
    t_hrs  = t_days * 24

    phase  = (t_hrs % period_hr) / period_hr
    y_true = amplitude * np.cos(4 * np.pi * phase)  # double hump
    y_obs  = y_true + rng.normal(0, noise, n_obs)
    dy     = np.full(n_obs, noise)

    # Assign bands
    band_choices = ["g", "r", "i"]
    bands = rng.choice(band_choices, size=n_obs)

    # Add per-band offsets (simulate colour)
    band_offsets = {"g": 0.3, "r": 0.0, "i": -0.15}
    for b, off in band_offsets.items():
        y_obs[bands == b] += off

    mjd = t_days + 60000.0  # arbitrary MJD epoch

    return pd.DataFrame({
        "provid":  "TEST0001",
        "mjd":     mjd,
        "band":    bands,
        "mag":     22.0 + y_obs,
        "rmsmag":  dy,
    })


# ── Tests: GLS ────────────────────────────────────────────────────────────────

class TestGLS:
    def test_gls_peak_at_true_period(self):
        """GLS should find a clear peak at the true period / 2 (single harmonic)."""
        df     = make_synthetic_lc(period_hr=4.8)
        df_    = df.copy(); df_["mag_corr"] = df_["mag"] - df_.groupby("band")["mag"].transform("median")
        t_hrs  = (df_["mjd"].values - df_["mjd"].min()) * 24
        y      = df_["mag_corr"].values
        dy     = df_["rmsmag"].values

        test_p = np.linspace(0.5, 12, 2000)
        power  = gls_periodogram(t_hrs, y, dy, test_p)
        best   = test_p[np.argmax(power)]

        # GLS finds P/2 = 2.4 hr for double-hump (expected behaviour)
        assert abs(best - 2.4) < 0.2 or abs(best - 4.8) < 0.2, \
            f"GLS best period {best:.3f} not near 2.4 or 4.8 hr"

    def test_gls_power_range(self):
        """GLS power should be in [0, 1]."""
        t  = np.linspace(0, 100, 50)
        y  = np.sin(2 * np.pi * t / 4.8)
        dy = np.ones(50) * 0.05
        pw = gls_power(t, y, dy, 4.8)
        assert 0.0 <= pw <= 1.0, f"GLS power {pw} out of [0,1]"

    def test_gls_flat_data_low_power(self):
        """Flat (constant) data should give near-zero GLS power."""
        t  = np.linspace(0, 100, 50)
        y  = np.zeros(50)
        dy = np.ones(50) * 0.05
        pw = gls_power(t, y, dy, 4.8)
        assert pw < 0.1, f"GLS power {pw:.3f} too high for flat data"


# ── Tests: MHAOV ─────────────────────────────────────────────────────────────

class TestMHAOV:
    def test_mhaov_high_F_at_true_period(self):
        """MHAOV F-statistic should be high at true period."""
        df    = make_synthetic_lc(period_hr=4.8, amplitude=0.4, noise=0.02)
        df_   = df.copy(); df_["mag_corr"] = df_["mag"] - df_.groupby("band")["mag"].transform("median")
        t_hrs = (df_["mjd"].values - df_["mjd"].min()) * 24
        y     = df_["mag_corr"].values
        dy    = df_["rmsmag"].values

        F_true  = mhaov_single(t_hrs, y, dy, 4.8, nh=2)
        F_wrong = mhaov_single(t_hrs, y, dy, 7.3, nh=2)
        assert F_true > F_wrong, f"F at true period ({F_true:.2f}) not > F at wrong period ({F_wrong:.2f})"

    def test_mhaov_nonnegative(self):
        """MHAOV F-statistic should always be non-negative."""
        rng = np.random.default_rng(0)
        t   = np.sort(rng.uniform(0, 100, 50))
        y   = rng.normal(0, 0.1, 50)
        dy  = np.ones(50) * 0.05
        F   = mhaov_single(t, y, dy, 3.5, nh=2)
        assert F >= 0.0, f"MHAOV F={F} is negative"


# ── Tests: Conditional Entropy ────────────────────────────────────────────────

class TestConditionalEntropy:
    def test_ce_lower_at_true_period(self):
        """CE should be lower (more structured) at true period."""
        df    = make_synthetic_lc(period_hr=4.8, amplitude=0.5, noise=0.02)
        df_   = df.copy(); df_["mag_corr"] = df_["mag"] - df_.groupby("band")["mag"].transform("median")
        t_hrs = (df_["mjd"].values - df_["mjd"].min()) * 24
        y     = df_["mag_corr"].values

        ce_true  = ce_single(t_hrs, y, 4.8)
        ce_wrong = ce_single(t_hrs, y, 7.1)
        assert ce_true <= ce_wrong, \
            f"CE at true period ({ce_true:.4f}) not <= CE at wrong ({ce_wrong:.4f})"


# ── Tests: Agreement check ────────────────────────────────────────────────────

class TestAgreementCheck:
    def test_agrees_within_tolerance(self):
        agrees, spread = check_agreement(4.80, 4.82, 4.79, tol=0.05)
        assert agrees

    def test_disagrees_outside_tolerance(self):
        agrees, spread = check_agreement(4.80, 5.50, 4.82, tol=0.05)
        assert not agrees

    def test_spread_computed_correctly(self):
        _, spread = check_agreement(4.0, 4.4, 4.2, tol=0.05)
        # median=4.2, max_dev = 0.2/4.2 ≈ 0.0476
        assert abs(spread - 0.2 / 4.2) < 0.001


# ── Tests: Preprocessing ─────────────────────────────────────────────────────

class TestPreprocessing:
    def test_band_offsets_removed(self):
        """After band offset correction, each band should have zero median."""
        df = make_synthetic_lc()
        df_corrected = _apply_band_offsets(df.copy(), DEFAULT_CONFIG)
        for band in df["band"].unique():
            median = df_corrected.loc[df_corrected["band"] == band, "mag_corr"].median()
            assert abs(median) < 1e-10, f"Band {band} median not zero after offset: {median:.6f}"

    def test_detrend_removes_polynomial(self):
        """Detrend should remove a linear trend."""
        t = np.linspace(0, 100, 100)
        y = 0.01 * t + np.sin(2 * np.pi * t / 5)  # linear trend + signal
        y_dt, _ = _geometry_detrend(t, y)
        # After detrend, correlation with t should be small
        corr = abs(np.corrcoef(t, y_dt)[0, 1])
        assert corr < 0.1, f"Residual correlation with time too high: {corr:.3f}"

    def test_snr_increases_with_amplitude(self):
        """Higher amplitude should give higher SNR."""
        t  = np.linspace(0, 50, 100)
        dy = np.ones(100) * 0.05
        y_low  = 0.1 * np.sin(2 * np.pi * t / 5)
        y_high = 0.5 * np.sin(2 * np.pi * t / 5)
        assert compute_snr(y_low, dy) < compute_snr(y_high, dy)

    def test_preprocess_returns_correct_provid(self):
        df = make_synthetic_lc()
        data = preprocess(df, DEFAULT_CONFIG)
        assert data.provid == "TEST0001"

    def test_preprocess_time_in_hours(self):
        """Preprocessed time should start at 0 and be in hours."""
        df = make_synthetic_lc()
        data = preprocess(df, DEFAULT_CONFIG)
        assert data.t_hrs[0] == pytest.approx(0.0, abs=1e-6)
        assert data.t_hrs[-1] > 0


# ── Tests: Bayesian ───────────────────────────────────────────────────────────

class TestBayesian:
    def test_posterior_sums_to_one(self):
        """Posterior should be normalised (integrates to ~1)."""
        df    = make_synthetic_lc(period_hr=4.8)
        df_   = df.copy(); df_["mag_corr"] = df_["mag"] - df_.groupby("band")["mag"].transform("median")
        t_hrs = (df_["mjd"].values - df_["mjd"].min()) * 24
        y     = df_["mag_corr"].values
        dy    = df_["rmsmag"].values

        tp   = np.linspace(0.5, 12, 1000)
        post = bayesian_period_posterior(t_hrs, y, dy, tp)
        norm = np.trapezoid(post, tp)
        assert abs(norm - 1.0) < 0.05, f"Posterior norm = {norm:.4f}, expected ~1.0"

    def test_credible_interval_contains_true_period(self):
        """95% CI should contain the true period for high-SNR synthetic data."""
        df    = make_synthetic_lc(period_hr=4.8, amplitude=0.5, noise=0.01, n_obs=200)
        df_   = df.copy(); df_["mag_corr"] = df_["mag"] - df_.groupby("band")["mag"].transform("median")
        t_hrs = (df_["mjd"].values - df_["mjd"].min()) * 24
        y     = df_["mag_corr"].values
        dy    = df_["rmsmag"].values

        tp    = np.linspace(1.0, 10.0, 3000)
        post  = bayesian_period_posterior(t_hrs, y, dy, tp)
        lo,hi = compute_credible_interval(post, tp)
        # True period is 4.8 or 2.4 (harmonic) — CI should cover one of them
        assert lo <= 4.8 <= hi or lo <= 2.4 <= hi, \
            f"95% CI [{lo:.3f},{hi:.3f}] does not contain true period 4.8 or 2.4"


# ── Tests: Catalog ────────────────────────────────────────────────────────────

class TestCatalog:
    def test_append_result_increases_rows(self):
        df   = make_synthetic_lc()
        data = preprocess(df, DEFAULT_CONFIG)
        from tier1 import run_tier1
        t1   = run_tier1(data, DEFAULT_CONFIG)
        row  = result_to_row(data, t1)

        catalog = init_catalog(DEFAULT_CONFIG)
        n_before = len(catalog)
        catalog  = append_result(catalog, row)
        assert len(catalog) == n_before + 1

    def test_upsert_replaces_existing(self):
        df   = make_synthetic_lc()
        data = preprocess(df, DEFAULT_CONFIG)
        from tier1 import run_tier1
        t1   = run_tier1(data, DEFAULT_CONFIG)
        row  = result_to_row(data, t1)

        catalog = init_catalog(DEFAULT_CONFIG)
        catalog = append_result(catalog, row)
        catalog = append_result(catalog, row)  # second insert = upsert
        assert len(catalog) == 1, "Duplicate provid should upsert, not duplicate"


# ── Run tests ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
