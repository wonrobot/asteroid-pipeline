"""
tests/test_change7.py
---------------------
Tests for Change 7: ZTF augmentation, source tagging, n_sources wiring.

Run with: pytest tests/test_change7.py -v

Coverage
--------
- ZTF _standardise(): column variant handling, band mapping, offset application
- merge_with_rubin(): source tagging, no-data passthrough, sort order
- ingestion._post_process(): source="Rubin" auto-tag, pre-existing tag preserved
- characterise(): n_sources wired from df["source"], combined regime unlocked
- _classify_regime(): n_sources > 1 → "combined" regardless of other criteria
- DataConfig: all new ZTF fields present with safe defaults
- pipeline._maybe_augment_ztf(): trigger logic (mocked), failure graceful
"""

import sys, os
# conftest.py handles sys.path — this is a belt-and-suspenders fallback
_src_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _rubin_df(n=60, seed=0):
    """Minimal Rubin-like observations DataFrame (no source column yet)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "provid":  ["TEST"] * n,
        "mjd":     np.sort(rng.uniform(60000, 60010, n)),
        "band":    rng.choice(["Lg", "Lr", "Li"], n),
        "mag":     22.0 + rng.normal(0, 0.1, n),
        "rmsmag":  np.full(n, 0.05),
    })


def _ztf_df(n=30, seed=1):
    """Minimal ZTF-tagged DataFrame as returned by fetch_ztf()."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "provid":  ["TEST"] * n,
        "mjd":     np.sort(rng.uniform(58000, 60000, n)),  # ZTF era
        "band":    rng.choice(["Lg", "Lr", "Li"], n),
        "mag":     22.0 + rng.normal(0, 0.12, n),
        "rmsmag":  np.full(n, 0.06),
        "source":  ["ZTF"] * n,
    })


# ── ZTF module: _standardise ──────────────────────────────────────────────────

class TestZTFStandardise:
    """Tests for the raw IRSA response → standard columns converter."""

    def _make_raw(self, mjd_col="mjd", mag_col="mag", err_col="magerr",
                  band_col="filtercode", band_val="zr"):
        return pd.DataFrame({
            mjd_col: [60000.0, 60001.5],
            mag_col: [22.1, 22.2],
            err_col: [0.03, 0.04],
            band_col: [band_val] * 2,
        })

    def test_standard_columns(self):
        """Standard IRSA column names are handled correctly."""
        from sources.ztf import _standardise
        raw = self._make_raw()
        out = _standardise(raw, "TEST", False)
        assert "mjd" in out.columns
        assert "mag" in out.columns
        assert "rmsmag" in out.columns
        assert "band" in out.columns
        assert "source" in out.columns
        assert out["source"].iloc[0] == "ZTF"

    def test_jd_converted_to_mjd(self):
        """JD column is converted to MJD (subtract 2400000.5)."""
        from sources.ztf import _standardise
        raw = self._make_raw(mjd_col="jd")
        raw["jd"] = raw["jd"] + 2400000.5   # make it a JD
        out = _standardise(raw, "TEST", False)
        assert abs(out["mjd"].iloc[0] - 60000.0) < 0.01

    def test_psfmag_columns(self):
        """Older IRSA releases use psfmag / psfmagerr."""
        from sources.ztf import _standardise
        raw = self._make_raw(mag_col="psfmag", err_col="psfmagerr",
                             band_col="bandname", band_val="zg")
        out = _standardise(raw, "TEST", False)
        assert list(out["band"].unique()) == ["Lg"]

    def test_filtercode_to_canonical_band(self):
        """IRSA filtercode (zg/zr/zi) maps to pipeline canonical (Lg/Lr/Li)."""
        from sources.ztf import _standardise, ZTF_BAND_MAP
        for ztf_code, canonical in ZTF_BAND_MAP.items():
            raw = self._make_raw(band_val=ztf_code)
            out = _standardise(raw, "TEST", False)
            assert out["band"].iloc[0] == canonical, \
                f"ZTF filtercode {ztf_code} → got {out['band'].iloc[0]}, expected {canonical}"

    def test_apply_offsets_shifts_magnitude(self):
        """apply_offsets=True should shift ZTF magnitudes by ZTF_RUBIN_OFFSETS."""
        from sources.ztf import _standardise, ZTF_RUBIN_OFFSETS
        raw = self._make_raw(band_val="zr")   # zr → Lr, offset=-0.01
        raw["mag"] = 22.0
        out_no   = _standardise(raw.copy(), "TEST", apply_offsets=False)
        out_yes  = _standardise(raw.copy(), "TEST", apply_offsets=True)
        expected_shift = ZTF_RUBIN_OFFSETS["Lr"]
        delta = float(out_yes["mag"].iloc[0] - out_no["mag"].iloc[0])
        assert abs(delta - expected_shift) < 1e-9, \
            f"Offset for Lr: got {delta}, expected {expected_shift}"

    def test_returns_empty_on_missing_time_column(self):
        """DataFrame with no time column returns empty."""
        from sources.ztf import _standardise
        raw = pd.DataFrame({"mag": [22.0], "magerr": [0.05], "filtercode": ["zr"]})
        out = _standardise(raw, "TEST", False)
        assert len(out) == 0

    def test_returns_empty_on_missing_mag_column(self):
        """DataFrame with no magnitude column returns empty."""
        from sources.ztf import _standardise
        raw = pd.DataFrame({"mjd": [60000.0], "magerr": [0.05], "filtercode": ["zr"]})
        out = _standardise(raw, "TEST", False)
        assert len(out) == 0

    def test_oid_column_preserved(self):
        """If 'oid' is present in raw data it is kept for deduplication."""
        from sources.ztf import _standardise
        raw = self._make_raw()
        raw["oid"] = [12345, 67890]
        out = _standardise(raw, "TEST", False)
        assert "oid" in out.columns


# ── ZTF module: merge_with_rubin ──────────────────────────────────────────────

class TestMergeWithRubin:
    def test_source_column_added_to_rubin(self):
        """Rubin rows get source='Rubin'."""
        from sources.ztf import merge_with_rubin
        df_r   = _rubin_df()
        df_ztf = _ztf_df()
        out    = merge_with_rubin(df_r, df_ztf)
        rubin_sources = out.loc[out["mjd"].between(60000, 60010), "source"].unique()
        assert "Rubin" in rubin_sources

    def test_ztf_source_preserved(self):
        """ZTF rows retain source='ZTF'."""
        from sources.ztf import merge_with_rubin
        df_r   = _rubin_df()
        df_ztf = _ztf_df()
        out    = merge_with_rubin(df_r, df_ztf)
        assert "ZTF" in out["source"].unique()

    def test_sorted_by_mjd(self):
        """Output is sorted by mjd."""
        from sources.ztf import merge_with_rubin
        df_r   = _rubin_df()
        df_ztf = _ztf_df()
        out    = merge_with_rubin(df_r, df_ztf)
        assert out["mjd"].is_monotonic_increasing, "Output not sorted by mjd"

    def test_row_count_correct(self):
        """Output has exactly len(df_r) + len(df_ztf) rows."""
        from sources.ztf import merge_with_rubin
        df_r   = _rubin_df(n=60)
        df_ztf = _ztf_df(n=30)
        out    = merge_with_rubin(df_r, df_ztf)
        assert len(out) == 90

    def test_empty_ztf_passthrough(self):
        """Empty ZTF DataFrame returns Rubin-only with source column."""
        from sources.ztf import merge_with_rubin, _empty_df
        df_r = _rubin_df()
        out  = merge_with_rubin(df_r, _empty_df())
        assert len(out) == len(df_r)
        assert list(out["source"].unique()) == ["Rubin"]

    def test_none_ztf_passthrough(self):
        """None ZTF argument is handled gracefully."""
        from sources.ztf import merge_with_rubin
        df_r = _rubin_df()
        out  = merge_with_rubin(df_r, None)
        assert len(out) == len(df_r)
        assert "source" in out.columns

    def test_apply_offsets_shifts_ztf_only(self):
        """apply_offsets=True shifts ZTF magnitudes, not Rubin magnitudes."""
        from sources.ztf import merge_with_rubin, ZTF_RUBIN_OFFSETS
        df_r   = _rubin_df(n=10, seed=0)
        df_ztf = _ztf_df(n=10, seed=1)
        df_ztf["band"] = "Lr"  # known offset = -0.01
        df_ztf["mag"]  = 22.0

        out_no  = merge_with_rubin(df_r.copy(), df_ztf.copy(), apply_offsets=False)
        out_yes = merge_with_rubin(df_r.copy(), df_ztf.copy(), apply_offsets=True)

        ztf_no  = out_no[out_no["source"]=="ZTF"]["mag"].mean()
        ztf_yes = out_yes[out_yes["source"]=="ZTF"]["mag"].mean()
        assert abs((ztf_yes - ztf_no) - ZTF_RUBIN_OFFSETS["Lr"]) < 1e-9

        rub_no  = out_no[out_no["source"]=="Rubin"]["mag"].mean()
        rub_yes = out_yes[out_yes["source"]=="Rubin"]["mag"].mean()
        assert abs(rub_no - rub_yes) < 1e-12, "Rubin magnitudes should not change"


# ── Ingestion: source tagging ─────────────────────────────────────────────────

class TestIngestionSourceTag:
    def test_post_process_adds_rubin_source(self):
        """_post_process adds source='Rubin' when column is absent."""
        from ingestion import _post_process
        from config import DEFAULT_CONFIG
        df  = _rubin_df()
        out = _post_process(df, DEFAULT_CONFIG)
        assert "source" in out.columns
        assert (out["source"] == "Rubin").all(), \
            f"Expected all Rubin, got {out['source'].unique()}"

    def test_post_process_preserves_existing_source(self):
        """_post_process does not overwrite a source column that already exists."""
        from ingestion import _post_process
        from config import DEFAULT_CONFIG
        df            = _rubin_df()
        df["source"]  = "Custom"
        out           = _post_process(df, DEFAULT_CONFIG)
        # Existing source column should be preserved (not overwritten to Rubin)
        assert "Custom" in out["source"].values, \
            "Pre-existing source column was overwritten"

    def test_source_column_in_list_objects_output(self):
        """list_objects does not crash when source column is present."""
        from ingestion import _post_process, list_objects
        from config import DEFAULT_CONFIG
        df  = _rubin_df(n=60)
        out = _post_process(df, DEFAULT_CONFIG)
        summary = list_objects(out)
        assert "provid" in summary.columns


# ── Characterise: n_sources wiring ───────────────────────────────────────────

class TestCharacteriseNSources:
    def _df_with_sources(self, n_rubin=60, n_ztf=30, seed=0):
        """Combined DataFrame with both source tags."""
        df_r = _rubin_df(n=n_rubin, seed=seed)
        df_r["source"] = "Rubin"
        df_z = _ztf_df(n=n_ztf, seed=seed+1)
        return pd.concat([df_r, df_z], ignore_index=True)

    def test_rubin_only_gives_nsources_1(self):
        """Rubin-only data → n_sources=1."""
        from characterise import characterise
        df   = _rubin_df()
        df["source"] = "Rubin"
        char = characterise(df)
        assert char.n_sources == 1
        assert char.sources == ["Rubin"]

    def test_combined_gives_nsources_2(self):
        """Rubin + ZTF data → n_sources=2."""
        from characterise import characterise
        df   = self._df_with_sources()
        char = characterise(df)
        assert char.n_sources == 2
        assert set(char.sources) == {"Rubin", "ZTF"}

    def test_combined_regime_unlocked(self):
        """n_sources=2 unlocks regime='combined' regardless of other criteria."""
        from characterise import characterise
        df   = self._df_with_sources()
        char = characterise(df)
        assert char.regime == "combined", \
            f"Expected combined, got {char.regime}"

    def test_combined_regime_has_high_ceiling(self):
        """combined regime has reliability_ceiling='high'."""
        from characterise import characterise
        df   = self._df_with_sources()
        char = characterise(df)
        # n_obs = 90 >= 30, so ceiling is not downgraded
        assert char.reliability_ceiling == "high"

    def test_combined_recommends_all_methods(self):
        """combined regime recommends MBLS, MHAOV, CE, CLEAN."""
        from characterise import characterise
        df   = self._df_with_sources()
        char = characterise(df)
        for method in ["mbls", "mhaov", "ce", "clean"]:
            assert method in char.recommended_methods, \
                f"Method {method} not in recommended_methods for combined regime"

    def test_no_source_column_defaults_to_1(self):
        """DataFrame without source column → n_sources=1 (backward compat)."""
        from characterise import characterise
        df   = _rubin_df()
        # No "source" column — old-style input
        assert "source" not in df.columns
        char = characterise(df)
        assert char.n_sources == 1
        assert char.sources == ["Rubin"]

    def test_sources_listed_in_notes(self):
        """Notes field mentions multiple sources when n_sources > 1."""
        from characterise import characterise
        df   = self._df_with_sources()
        char = characterise(df)
        assert "Sources=2" in char.notes or "ZTF" in char.notes, \
            f"Expected source count in notes, got: {char.notes}"


# ── _classify_regime: combined trigger ───────────────────────────────────────

class TestClassifyRegimeCombined:
    def test_single_source_does_not_trigger_combined(self):
        from characterise import _classify_regime
        r = _classify_regime(
            n_obs=100, baseline_days=15, n_nights=10, n_seasons=1,
            obs_per_night_median=10, n_sources=1
        )
        assert r != "combined"

    def test_two_sources_triggers_combined(self):
        from characterise import _classify_regime
        r = _classify_regime(
            n_obs=100, baseline_days=15, n_nights=10, n_seasons=1,
            obs_per_night_median=10, n_sources=2
        )
        assert r == "combined"

    def test_three_sources_triggers_combined(self):
        from characterise import _classify_regime
        r = _classify_regime(
            n_obs=200, baseline_days=1000, n_nights=100, n_seasons=5,
            obs_per_night_median=2, n_sources=3
        )
        assert r == "combined"

    def test_combined_overrides_sparse(self):
        """n_sources > 1 wins over sparse classification criteria."""
        from characterise import _classify_regime
        r = _classify_regime(
            n_obs=25, baseline_days=5, n_nights=3, n_seasons=1,
            obs_per_night_median=8, n_sources=2
        )
        assert r == "combined"

    def test_combined_overrides_rich_multiyear(self):
        """n_sources > 1 wins even over rich_multiyear criteria."""
        from characterise import _classify_regime
        r = _classify_regime(
            n_obs=600, baseline_days=800, n_nights=200, n_seasons=4,
            obs_per_night_median=3, n_sources=2
        )
        assert r == "combined"


# ── DataConfig: ZTF fields ────────────────────────────────────────────────────

class TestDataConfigZTF:
    def test_all_ztf_fields_present(self):
        """All Change 7 ZTF fields must be present in DataConfig."""
        from config import DataConfig
        dc = DataConfig()
        fields = [
            "use_ztf", "ztf_search_radius_arcsec", "ztf_n_ephemeris_points",
            "ztf_time_window_days", "ztf_min_obs", "ztf_trigger_n_obs",
            "ztf_apply_offsets", "ztf_date_start",
        ]
        for f in fields:
            assert hasattr(dc, f), f"DataConfig missing field: {f}"

    def test_use_ztf_defaults_false(self):
        """use_ztf must default to False — safe for bulk offline runs."""
        from config import DataConfig
        assert DataConfig().use_ztf is False

    def test_ztf_apply_offsets_defaults_false(self):
        """ztf_apply_offsets defaults to False — MBLS handles it internally."""
        from config import DataConfig
        assert DataConfig().ztf_apply_offsets is False

    def test_ztf_date_start_is_ztf_era(self):
        """ztf_date_start must be on or after ZTF survey start (2018-03-01)."""
        from config import DataConfig
        assert DataConfig().ztf_date_start >= "2018-03-01"

    def test_ztf_min_obs_positive(self):
        from config import DataConfig
        assert DataConfig().ztf_min_obs > 0

    def test_use_atlas_field_present(self):
        """use_atlas stub field must be present (Change 7 Phase 4 placeholder)."""
        from config import DataConfig
        assert hasattr(DataConfig(), "use_atlas")


# ── Pipeline: _maybe_augment_ztf ─────────────────────────────────────────────

class TestMaybeAugmentZTF:
    """
    Tests for _maybe_augment_ztf() trigger logic using mocked fetch_ztf.
    We never actually call IRSA — the goal is to verify the gate conditions.
    """

    def _char(self, regime="sparse", n_obs=30, n_sources=1):
        """Return a minimal DataCharacterisation with given properties."""
        from characterise import DataCharacterisation
        return DataCharacterisation(
            provid="TEST", regime=regime, n_obs=n_obs, n_bands=3,
            bands=["Lg","Lr","Li"], baseline_days=10.0, n_nights=5,
            n_seasons=1, obs_per_night_median=6.0, obs_per_night_max=10,
            night_duration_hr=3.0, snr_proxy=5.0, mag_range=0.3,
            n_sources=n_sources, sources=["Rubin"],
            dominant_aliases_hr=[12.,24.], alias_risk={},
            lcdb_period=float("nan"), lcdb_u_code=0,
            recommended_methods=["mbls","mhaov"],
            reliability_ceiling="medium", notes="test",
        )

    def _config(self, use_ztf=True, ztf_min_obs=10, trigger_n_obs=40):
        from config import PipelineConfig, DataConfig
        dc = DataConfig(use_ztf=use_ztf, ztf_min_obs=ztf_min_obs,
                        ztf_trigger_n_obs=trigger_n_obs)
        return PipelineConfig(data=dc)

    def test_use_ztf_false_skips_entirely(self):
        """use_ztf=False → df_obj returned unchanged, no fetch attempted."""
        from pipeline import _maybe_augment_ztf
        df   = _rubin_df(n=25)
        char = self._char(regime="sparse")
        conf = self._config(use_ztf=False)
        out  = _maybe_augment_ztf(df, char, conf, "TEST")
        assert "source" not in out.columns or list(out.get("source", pd.Series([])).unique()) == ["Rubin"]
        assert len(out) == len(df)

    def test_combined_regime_skips(self):
        """Already combined regime → skip augmentation."""
        from pipeline import _maybe_augment_ztf
        df   = _rubin_df(n=60)
        char = self._char(regime="combined", n_sources=2)
        conf = self._config(use_ztf=True)
        out  = _maybe_augment_ztf(df, char, conf, "TEST")
        assert len(out) == len(df)

    def test_dense_high_obs_skips(self):
        """Dense regime + high n_obs → skip (doesn't need augmentation)."""
        from pipeline import _maybe_augment_ztf
        df   = _rubin_df(n=100)
        char = self._char(regime="dense", n_obs=100)
        conf = self._config(use_ztf=True, trigger_n_obs=40)
        out  = _maybe_augment_ztf(df, char, conf, "TEST")
        # n_obs=100 > trigger_n_obs=40 and regime != sparse → skip
        assert len(out) == len(df)

    def test_fetch_failure_returns_original(self, monkeypatch):
        """If fetch_ztf raises, original df is returned (graceful degradation)."""
        from pipeline import _maybe_augment_ztf
        import sources.ztf as ztf_mod

        def _fail(*a, **kw):
            raise RuntimeError("IRSA unreachable")

        monkeypatch.setattr(ztf_mod, "fetch_ztf", _fail)

        df   = _rubin_df(n=25)
        char = self._char(regime="sparse", n_obs=25)
        conf = self._config(use_ztf=True)
        out  = _maybe_augment_ztf(df, char, conf, "TEST")
        assert len(out) == len(df)

    def test_insufficient_ztf_returns_original(self, monkeypatch):
        """If ZTF returns too few observations, original df is returned."""
        from pipeline import _maybe_augment_ztf
        import sources.ztf as ztf_mod

        # Mock returns only 3 observations — below ztf_min_obs=10
        monkeypatch.setattr(ztf_mod, "fetch_ztf",
                            lambda *a, **kw: _ztf_df(n=3))

        df   = _rubin_df(n=25)
        df["source"] = "Rubin"
        char = self._char(regime="sparse", n_obs=25)
        conf = self._config(use_ztf=True, ztf_min_obs=10)
        out  = _maybe_augment_ztf(df, char, conf, "TEST")
        assert len(out) == 25   # unchanged

    def test_successful_augmentation_adds_ztf(self, monkeypatch):
        """Successful ZTF fetch merges data and adds source='ZTF' rows."""
        from pipeline import _maybe_augment_ztf
        import sources.ztf as ztf_mod

        df_ztf_mock = _ztf_df(n=20)
        monkeypatch.setattr(ztf_mod, "fetch_ztf",
                            lambda *a, **kw: df_ztf_mock)

        df   = _rubin_df(n=25)
        df["source"] = "Rubin"
        char = self._char(regime="sparse", n_obs=25)
        conf = self._config(use_ztf=True, ztf_min_obs=10)
        out  = _maybe_augment_ztf(df, char, conf, "TEST")

        assert len(out) == 45   # 25 Rubin + 20 ZTF
        assert "ZTF" in out["source"].unique()
        assert "Rubin" in out["source"].unique()

    def test_post_augment_characterise_gives_combined(self, monkeypatch):
        """After augmentation, characterise() sees regime='combined'."""
        from pipeline import _maybe_augment_ztf
        from characterise import characterise
        import sources.ztf as ztf_mod

        df_ztf_mock = _ztf_df(n=25)
        monkeypatch.setattr(ztf_mod, "fetch_ztf",
                            lambda *a, **kw: df_ztf_mock)

        df   = _rubin_df(n=30)
        df["source"] = "Rubin"
        char = self._char(regime="sparse", n_obs=30)
        conf = self._config(use_ztf=True, ztf_min_obs=10)
        out  = _maybe_augment_ztf(df, char, conf, "TEST")

        char_post = characterise(out)
        assert char_post.regime == "combined"
        assert char_post.n_sources == 2


# ── End-to-end smoke test ─────────────────────────────────────────────────────

class TestChange7EndToEnd:
    """
    Run the pipeline on a synthetic combined dataset to verify all components
    cooperate correctly. Does not require network access (no ZTF fetch).
    """

    def _require_pipeline(self):
        """Skip if core pipeline modules not available (incomplete checkout)."""
        try:
            import preprocessing  # noqa
        except ImportError:
            pytest.skip("Full pipeline modules not available in this environment")


    def _make_combined_df(self, n_rubin=80, n_ztf=40, period_hr=4.8,
                          amplitude=0.4, seed=0):
        """Synthetic combined Rubin+ZTF lightcurve with known period."""
        rng = np.random.default_rng(seed)

        def _lc(n, t_lo, t_hi, source_name):
            t = np.sort(rng.uniform(t_lo, t_hi, n))
            b = rng.choice(["Lg", "Lr", "Li"], n)
            offsets = {"Lg": 0.3, "Lr": 0.0, "Li": -0.15}
            phase = (t * 24 % period_hr) / period_hr
            mag = np.array([
                22.0 + offsets[bi] + amplitude * np.sin(4 * np.pi * ph)
                + rng.normal(0, 0.04)
                for bi, ph in zip(b, phase)
            ])
            return pd.DataFrame({
                "provid": ["SYNTH"] * n,
                "mjd": t,
                "band": b,
                "mag": mag,
                "rmsmag": np.full(n, 0.04),
                "source": [source_name] * n,
            })

        df_r = _lc(n_rubin, 60000, 60012, "Rubin")
        df_z = _lc(n_ztf,  58500, 59500, "ZTF")    # ZTF era baseline
        return pd.concat([df_r, df_z], ignore_index=True).sort_values("mjd").reset_index(drop=True)

    def test_characterise_sees_combined(self):
        self._require_pipeline()
        from characterise import characterise
        df = self._make_combined_df()
        char = characterise(df)
        assert char.regime == "combined"
        assert char.n_sources == 2

    def test_preprocessing_handles_combined(self):
        self._require_pipeline()
        from preprocessing import preprocess
        from config import DEFAULT_CONFIG
        df   = self._make_combined_df()
        data = preprocess(df, DEFAULT_CONFIG)
        assert data.n_obs > 0
        assert data.baseline_hr > 0

    def test_tier1_runs_on_combined(self):
        self._require_pipeline()
        from preprocessing import preprocess
        from tier1 import run_tier1
        from config import DEFAULT_CONFIG
        df   = self._make_combined_df(n_rubin=80, n_ztf=40, amplitude=0.5)
        data = preprocess(df, DEFAULT_CONFIG)
        t1   = run_tier1(data, DEFAULT_CONFIG)
        # With a clear signal, Tier 1 should pass
        assert hasattr(t1, "passes")
        assert hasattr(t1, "window_power")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
