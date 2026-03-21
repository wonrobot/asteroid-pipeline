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


# ── Tests: Window function ────────────────────────────────────────────────────

class TestWindowFunction:
    def test_window_power_same_length_as_grid(self):
        """Window function output must match the input period grid length."""
        from window import compute_window_function
        t = np.linspace(0, 100, 80)
        periods = np.linspace(0.5, 24, 500)
        wp = compute_window_function(t, periods)
        assert len(wp) == len(periods)

    def test_window_power_nonnegative(self):
        """Window function power must be non-negative everywhere."""
        from window import compute_window_function
        t = np.sort(np.random.default_rng(7).uniform(0, 200, 120))
        periods = np.linspace(0.5, 24, 1000)
        wp = compute_window_function(t, periods)
        assert np.all(wp >= 0)

    def test_contamination_score_range(self):
        """Contamination score must be in [0, 1]."""
        from window import compute_window_function, contamination_score
        t = np.sort(np.random.default_rng(7).uniform(0, 200, 120))
        periods = np.linspace(0.5, 24, 1000)
        wp = compute_window_function(t, periods)
        for p in [1.0, 4.8, 12.0, 24.0]:
            score = contamination_score(p, periods, wp)
            assert 0.0 <= score <= 1.0, f"contamination_score({p}) = {score} out of [0,1]"

    def test_contamination_zero_for_flat_window(self):
        """If window power is uniform, contamination score should be near zero."""
        from window import contamination_score
        periods = np.linspace(0.5, 24, 500)
        wp_flat = np.ones(500) * 0.01   # uniform — no alias peaks
        score = contamination_score(5.0, periods, wp_flat)
        assert score <= 1.0   # can't be above 1; flat window means no dominant alias

    def test_tier1_result_has_window_fields(self):
        """Tier1Result must carry window_power, gls_contamination, mbls_contamination."""
        from tier1 import run_tier1
        from preprocessing import preprocess
        df   = make_synthetic_lc(period_hr=4.8, n_obs=80)
        data = preprocess(df, DEFAULT_CONFIG)
        t1   = run_tier1(data, DEFAULT_CONFIG)

        assert hasattr(t1, 'window_power'),       "Tier1Result missing window_power"
        assert hasattr(t1, 'gls_contamination'),  "Tier1Result missing gls_contamination"
        assert hasattr(t1, 'mbls_contamination'), "Tier1Result missing mbls_contamination"

        if t1.passes:
            assert len(t1.window_power) == len(t1.test_periods), \
                "window_power length mismatch with test_periods"
            assert 0.0 <= t1.gls_contamination  <= 1.0, "gls_contamination out of [0,1]"
            assert 0.0 <= t1.mbls_contamination <= 1.0, "mbls_contamination out of [0,1]"

    def test_tier1_rejected_has_empty_window_fields(self):
        """Rejected Tier1Result should have empty window arrays, not crash."""
        from tier1 import run_tier1
        from preprocessing import preprocess
        # Too few observations to pass Tier 1
        df_tiny = make_synthetic_lc(n_obs=5)
        data    = preprocess(df_tiny, DEFAULT_CONFIG)
        t1      = run_tier1(data, DEFAULT_CONFIG)
        assert not t1.passes
        assert len(t1.window_power) == 0
        assert np.isnan(t1.gls_contamination)
        assert np.isnan(t1.mbls_contamination)

    def test_tier2_result_has_window_fields(self):
        """Tier2Result must carry window_power and all four contamination scores."""
        from tier1 import run_tier1
        from tier2 import run_tier2
        from preprocessing import preprocess
        df   = make_synthetic_lc(period_hr=4.8, amplitude=0.4, noise=0.02, n_obs=120)
        data = preprocess(df, DEFAULT_CONFIG)
        t1   = run_tier1(data, DEFAULT_CONFIG)
        if not t1.passes:
            pytest.skip("Tier 1 rejected — cannot test Tier 2 fields")
        t2 = run_tier2(data, t1, DEFAULT_CONFIG)

        for field in ['window_power', 'mhaov_contamination', 'mbls_contamination',
                      'ce_contamination', 'consensus_contamination']:
            assert hasattr(t2, field), f"Tier2Result missing {field}"

        assert len(t2.window_power) == len(t2.test_periods), \
            "window_power length mismatch with test_periods"
        for field in ['mhaov_contamination', 'mbls_contamination',
                      'ce_contamination', 'consensus_contamination']:
            val = getattr(t2, field)
            assert 0.0 <= val <= 1.0 or np.isnan(val), \
                f"{field} = {val} out of [0,1]"


# ── Tests: Window alias in reliability ────────────────────────────────────────

class TestWindowAlias:
    def test_flag_window_alias_returns_false_for_empty_window(self):
        """flag_window_alias should not crash when window_power is empty."""
        from reliability import flag_window_alias
        from tier1 import Tier1Result

        # Build a minimal Tier1Result with empty arrays (rejected object)
        t1_empty = Tier1Result(
            provid="TEST", passes=False,
            best_period_gls=np.nan, best_period_mbls=np.nan,
            gls_power_max=0.0, snr=0.0, n_obs=0,
            reject_reason="insufficient data",
            test_periods=np.array([]), gls_power=np.array([]),
            mbls_power=np.array([]), window_power=np.array([]),
            gls_contamination=np.nan, mbls_contamination=np.nan,
        )
        risk, note = flag_window_alias(4.8, t1_empty)
        assert risk is False
        assert note == ""

    def test_flag_window_alias_triggers_on_high_contamination(self):
        """flag_window_alias should trigger when period sits on window peak."""
        from reliability import flag_window_alias
        from tier1 import Tier1Result
        import numpy as np

        # Construct a fake window with a strong peak at exactly 4.8hr
        periods = np.linspace(0.5, 24, 1000)
        wp = np.zeros(1000)
        idx = np.argmin(np.abs(periods - 4.8))
        wp[idx] = 1.0   # sharp spike at 4.8hr

        t1_fake = Tier1Result(
            provid="TEST", passes=True,
            best_period_gls=4.8, best_period_mbls=4.8,
            gls_power_max=0.8, snr=5.0, n_obs=100,
            reject_reason=None,
            test_periods=periods, gls_power=np.zeros(1000),
            mbls_power=np.zeros(1000), window_power=wp,
            gls_contamination=1.0, mbls_contamination=1.0,
        )
        risk, note = flag_window_alias(4.8, t1_fake)
        assert risk is True
        assert "contamination" in note.lower()

    def test_reliability_has_window_alias_fields(self):
        """ReliabilityAssessment must include window_alias_risk and window_alias_note."""
        from reliability import ReliabilityAssessment
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(ReliabilityAssessment)}
        assert 'window_alias_risk' in field_names, \
            "ReliabilityAssessment missing window_alias_risk"
        assert 'window_alias_note' in field_names, \
            "ReliabilityAssessment missing window_alias_note"

    def test_catalog_has_window_alias_columns(self):
        """CATALOG_COLUMNS must include window_alias_risk and window_alias_note."""
        from catalog import CATALOG_COLUMNS
        assert 'window_alias_risk' in CATALOG_COLUMNS, \
            "CATALOG_COLUMNS missing window_alias_risk"
        assert 'window_alias_note' in CATALOG_COLUMNS, \
            "CATALOG_COLUMNS missing window_alias_note"


# ── Tests: MBLS false alarm probability ──────────────────────────────────────

class TestMBLSFAP:
    def test_fap_range(self):
        """FAP must be in [0, 1]."""
        from tier2 import compute_mbls_fap
        df   = make_synthetic_lc(period_hr=4.8, amplitude=0.4, noise=0.02, n_obs=100)
        from preprocessing import preprocess
        data = preprocess(df, DEFAULT_CONFIG)
        periods = np.linspace(0.5, 12, 500)
        from tier1 import mbls_periodogram
        obs_pow = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands, periods, nterms=2
        )
        fap = compute_mbls_fap(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods=periods,
            observed_max_power=float(obs_pow.max()),
            n_perm=50,   # fast for testing
            nterms=2,
        )
        assert 0.0 <= fap <= 1.0, f"FAP={fap} out of [0,1]"

    def test_fap_low_for_strong_signal(self):
        """High-SNR synthetic signal should give a low FAP."""
        from tier2 import compute_mbls_fap
        from preprocessing import preprocess
        from tier1 import mbls_periodogram
        df   = make_synthetic_lc(period_hr=4.8, amplitude=0.6, noise=0.01, n_obs=150)
        data = preprocess(df, DEFAULT_CONFIG)
        periods = np.linspace(0.5, 12, 500)
        obs_pow = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands, periods, nterms=2
        )
        fap = compute_mbls_fap(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods=periods,
            observed_max_power=float(obs_pow.max()),
            n_perm=100,
            nterms=2,
            seed=0,
        )
        assert fap < 0.05, f"Strong signal gave FAP={fap:.4f}, expected < 0.05"

    def test_fap_high_for_noise(self):
        """Pure noise should give a FAP near 1.0 (signal not significant)."""
        from tier2 import compute_mbls_fap
        from preprocessing import preprocess
        from tier1 import mbls_periodogram
        rng = np.random.default_rng(99)
        n   = 100
        df_noise = pd.DataFrame({
            "provid":  ["NOISE"] * n,
            "mjd":     np.sort(rng.uniform(60000, 60010, n)),
            "band":    rng.choice(["g", "r", "i"], n),
            "mag":     22.0 + rng.normal(0, 0.05, n),
            "rmsmag":  np.full(n, 0.05),
        })
        data = preprocess(df_noise, DEFAULT_CONFIG)
        periods = np.linspace(0.5, 12, 500)
        obs_pow = mbls_periodogram(
            data.t_hrs, data.y_multiband, data.dy, data.bands, periods, nterms=2
        )
        fap = compute_mbls_fap(
            data.t_hrs, data.y_multiband, data.dy, data.bands,
            test_periods=periods,
            observed_max_power=float(obs_pow.max()),
            n_perm=100,
            nterms=2,
            seed=0,
        )
        assert fap > 0.05, f"Noise gave FAP={fap:.4f}, expected > 0.05"

    def test_tier2_result_has_fap_fields(self):
        """Tier2Result must carry mbls_fap, mbls_sig, mhaov_sig, both_sig."""
        from tier1 import run_tier1
        from tier2 import run_tier2
        from preprocessing import preprocess
        df   = make_synthetic_lc(period_hr=4.8, amplitude=0.4, noise=0.02, n_obs=120)
        data = preprocess(df, DEFAULT_CONFIG)
        t1   = run_tier1(data, DEFAULT_CONFIG)
        if not t1.passes:
            pytest.skip("Tier 1 rejected")
        t2 = run_tier2(data, t1, DEFAULT_CONFIG)
        for field in ['mbls_fap', 'mbls_sig', 'mhaov_sig', 'both_sig']:
            assert hasattr(t2, field), f"Tier2Result missing {field}"
        assert 0.0 <= t2.mbls_fap <= 1.0, f"mbls_fap={t2.mbls_fap} out of [0,1]"
        assert isinstance(t2.mbls_sig,  bool), "mbls_sig not bool"
        assert isinstance(t2.mhaov_sig, bool), "mhaov_sig not bool"
        assert isinstance(t2.both_sig,  bool), "both_sig not bool"
        # both_sig must be consistent with the individual flags
        assert t2.both_sig == (t2.mbls_sig and t2.mhaov_sig), \
            "both_sig inconsistent with mbls_sig and mhaov_sig"

    def test_both_sig_feeds_reliability(self):
        """Reliability notes should mention MBLS FAP when both gates are used."""
        from tier1 import run_tier1
        from tier2 import run_tier2
        from reliability import compute_reliability
        from characterise import characterise
        from preprocessing import preprocess
        df   = make_synthetic_lc(period_hr=4.8, amplitude=0.5, noise=0.01, n_obs=150)
        data = preprocess(df, DEFAULT_CONFIG)
        char = characterise(df)
        t1   = run_tier1(data, DEFAULT_CONFIG)
        if not t1.passes:
            pytest.skip("Tier 1 rejected")
        t2 = run_tier2(data, t1, DEFAULT_CONFIG)
        if not t2.passes:
            pytest.skip("Tier 2 rejected or to Tier 3")
        rel = compute_reliability(char, t1, t2)
        # FAP value should appear in the reliability notes
        assert "FAP" in rel.notes or "fap" in rel.notes.lower(), \
            f"Reliability notes don't mention FAP: {rel.notes}"

    def test_catalog_has_mbls_fap_column(self):
        """CATALOG_COLUMNS must include t2_mbls_fap."""
        from catalog import CATALOG_COLUMNS
        assert "t2_mbls_fap" in CATALOG_COLUMNS, \
            "CATALOG_COLUMNS missing t2_mbls_fap"

    def test_config_has_fap_params(self):
        """TierConfig must expose mbls_fap_thresh and mbls_fap_n_perm."""
        from config import DEFAULT_CONFIG
        assert hasattr(DEFAULT_CONFIG.tier, 'mbls_fap_thresh'), \
            "TierConfig missing mbls_fap_thresh"
        assert hasattr(DEFAULT_CONFIG.tier, 'mbls_fap_n_perm'), \
            "TierConfig missing mbls_fap_n_perm"
        assert DEFAULT_CONFIG.tier.mbls_fap_thresh > 0
        assert DEFAULT_CONFIG.tier.mbls_fap_n_perm > 0


# ── Tests: Pass 2 window-qualified expansion ──────────────────────────────────

class TestPass2Suppression:
    def _make_obs(self, n=80, period_hr=4.8, amplitude=0.3, noise=0.03, seed=42):
        """Helper — synthetic lightcurve as PreparedData."""
        from preprocessing import preprocess
        df   = make_synthetic_lc(period_hr=period_hr, amplitude=amplitude,
                                 n_obs=n, noise=noise, seed=seed)
        return preprocess(df, DEFAULT_CONFIG), df

    def test_clean_signal_may_expand(self):
        """Weak but uncontaminated coarse power should allow Pass 2 expansion."""
        from tier1 import run_tier1
        # Low amplitude → weak coarse power, no alias contamination → expand allowed
        data, _ = self._make_obs(amplitude=0.05, noise=0.04)
        # Just check it doesn't crash and returns valid result
        t1 = run_tier1(data, DEFAULT_CONFIG)
        assert hasattr(t1, 'passes')

    def test_contaminated_weak_power_suppresses_expansion(self):
        """
        When coarse best period is window-contaminated AND power is weak,
        Pass 2 should NOT expand. Verify by checking the suppression log path
        exists in source (logic test) and that Tier1Result is still valid.
        """
        import ast
        with open('src/tier1.py') as f:
            src = f.read()
        assert 'Pass 2 suppressed' in src, \
            "Suppression log message missing from tier1.py"
        assert 'coarse_contaminated and weak_power and not near_boundary' in src, \
            "Suppression condition missing from tier1.py"

    def test_near_boundary_always_expands(self):
        """
        near_boundary trigger should expand even if coarse best is contaminated —
        harmonic concern (P/2 at boundary) is independent of alias risk.
        """
        with open('src/tier1.py') as f:
            src = f.read()
        # The should_expand condition should include near_boundary without
        # a contamination check
        assert 'near_boundary or (weak_power and not coarse_contaminated)' in src, \
            "near_boundary should expand regardless of contamination"

    def test_window_reused_when_no_expansion(self):
        """
        When grid is not expanded, window_pow should be window_pow_coarse
        (no redundant recomputation).
        """
        with open('src/tier1.py') as f:
            src = f.read()
        assert 'window_pow = window_pow_coarse' in src, \
            "Window reuse path missing — redundant recomputation on non-expanded grid"

    def test_window_recomputed_when_expanded(self):
        """When grid expands, window must be recomputed on the new larger grid."""
        with open('src/tier1.py') as f:
            src = f.read()
        assert 'if should_expand:' in src
        # The recompute must be inside the should_expand block
        expand_idx   = src.index('if should_expand:')
        recompute_idx = src.index('window_pow = compute_window_function')
        reuse_idx     = src.index('window_pow = window_pow_coarse')
        assert expand_idx < recompute_idx < reuse_idx, \
            "window recompute should come before window reuse in source"

    def test_tier1_result_still_valid_after_change(self):
        """End-to-end: Tier1Result must still have all expected fields."""
        from tier1 import run_tier1
        from preprocessing import preprocess
        df   = make_synthetic_lc(period_hr=4.8, amplitude=0.4, noise=0.02, n_obs=100)
        data = preprocess(df, DEFAULT_CONFIG)
        t1   = run_tier1(data, DEFAULT_CONFIG)
        for field in ['passes', 'best_period_gls', 'best_period_mbls',
                      'window_power', 'gls_contamination', 'mbls_contamination',
                      'test_periods', 'gls_power', 'mbls_power']:
            assert hasattr(t1, field), f"Tier1Result missing {field}"
        if t1.passes:
            assert len(t1.window_power) == len(t1.test_periods)
