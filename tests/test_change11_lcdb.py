"""
tests/test_change11_lcdb.py
---------------------------
Tests for Change 11: LCDB wiring through run_single_asteroid().

Covers
------
- _load_lcdb_records() returns empty dict gracefully on load failure
- run_single_asteroid() accepts lcdb_record=None (backward compat)
- run_single_asteroid() with a mock LCDBRecord: char, reliability, and
  catalog row all carry lcdb_period / lcdb_u_code / lcdb_agreement
- lcdb_agreement populated for exact, half_period, double_period, disagree
- No regression: run_single_asteroid() without lcdb_record still works
- Parallel worker _worker_call passes record from _worker_lcdb_records
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from config import PipelineConfig, DEFAULT_CONFIG
from sources.lcdb import LCDBRecord


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_lcdb_record(provid="2025 MA19", period_hr=8.879, u_code=3,
                      found=True, u_flag="3"):
    return LCDBRecord(
        provid=provid, number=None, name=None,
        period_hr=period_hr, u_code=u_code, u_flag=u_flag,
        amp_min=0.3, amp_max=1.0, taxonomy="S",
        hg_slope_G=0.15, is_binary=False, found=found,
    )


def _make_synthetic_lc(
    provid:    str   = "2025 MA19",
    period_hr: float = 8.879,
    n_obs:     int   = 120,
    amplitude: float = 0.4,
    noise:     float = 0.02,
    seed:      int   = 42,
) -> pd.DataFrame:
    """Minimal synthetic lightcurve suitable for pipeline ingestion."""
    rng = np.random.default_rng(seed)
    t   = np.sort(rng.uniform(0, 360, n_obs))  # hours
    mag = 18.0 + amplitude * np.sin(2 * np.pi * t / period_hr) + rng.normal(0, noise, n_obs)
    bands = rng.choice(["Lg", "Lr", "Li"], n_obs)
    return pd.DataFrame({
        "provid":  provid,
        "mjd":     t / 24.0,   # convert to days
        "mag":     mag,
        "rmsmag":  np.full(n_obs, noise * 2),
        "band":    bands,
        "source":  "Rubin",
    })


# ── _load_lcdb_records ────────────────────────────────────────────────────────

class TestLoadLcdbRecords:
    def test_graceful_on_load_failure(self):
        """If load_lcdb raises, _load_lcdb_records returns {} without crashing."""
        from pipeline import _load_lcdb_records
        with patch("pipeline.load_lcdb", side_effect=FileNotFoundError("no cache")):
            result = _load_lcdb_records(["2025 MA19", "2025 MB20"])
        assert result == {}

    def test_returns_dict_of_records(self):
        """Happy path: returns {provid: LCDBRecord} for all requested provids."""
        from pipeline import _load_lcdb_records
        rec_a = _make_lcdb_record("2025 MA19", period_hr=8.879)
        rec_b = LCDBRecord("2025 MB20", None, None, float("nan"), 0, "", float("nan"),
                           float("nan"), None, float("nan"), False, False)

        mock_df = MagicMock()
        with patch("pipeline.load_lcdb", return_value=mock_df), \
             patch("pipeline.lookup_batch",
                   return_value={"2025 MA19": rec_a, "2025 MB20": rec_b}):
            result = _load_lcdb_records(["2025 MA19", "2025 MB20"])

        assert result["2025 MA19"].found is True
        assert result["2025 MA19"].period_hr == pytest.approx(8.879)
        assert result["2025 MB20"].found is False

    def test_logs_found_count(self, caplog):
        """Info log must report n_found and u≥2 count."""
        import logging
        from pipeline import _load_lcdb_records
        rec_a = _make_lcdb_record("2025 MA19", period_hr=8.879, u_code=3)
        rec_b = LCDBRecord("2025 MB20", None, None, float("nan"), 0, "", float("nan"),
                           float("nan"), None, float("nan"), False, False)

        mock_df = MagicMock()
        with patch("pipeline.load_lcdb", return_value=mock_df), \
             patch("pipeline.lookup_batch",
                   return_value={"2025 MA19": rec_a, "2025 MB20": rec_b}), \
             caplog.at_level(logging.INFO, logger="pipeline"):
            _load_lcdb_records(["2025 MA19", "2025 MB20"])

        assert any("LCDB" in r.message for r in caplog.records)


# ── run_single_asteroid lcdb_record threading ─────────────────────────────────

class TestRunSingleAsteroidLcdb:
    """Integration-level: run_single_asteroid with a synthetic LC + mock LCDB."""

    def _run(self, provid="2025 MA19", period_hr=8.879, lcdb_record=None):
        from pipeline import run_single_asteroid
        df = _make_synthetic_lc(provid=provid, period_hr=period_hr)
        cfg = PipelineConfig()
        cfg.tier.lrt_n_objects = 0   # single object — raw alpha=0.05
        return run_single_asteroid(df, cfg, lcdb_record=lcdb_record)

    def test_no_lcdb_record_backward_compat(self):
        """Passing no lcdb_record must not error and lcdb fields are NaN/no_prior."""
        row = self._run(lcdb_record=None)
        assert row["lcdb_agreement"] in ("no_prior", None, "")
        assert math.isnan(row["lcdb_period_hr"] or float("nan")) or \
               row["lcdb_period_hr"] is None

    def test_not_found_record_backward_compat(self):
        """LCDBRecord with found=False behaves like no record."""
        rec = _make_lcdb_record(found=False)
        row = self._run(lcdb_record=rec)
        assert row["lcdb_agreement"] in ("no_prior", None, "")

    def test_found_record_populates_char_columns(self):
        """When LCDB record is found, lcdb_period_hr and lcdb_u_code appear in row."""
        rec = _make_lcdb_record(period_hr=8.879, u_code=3)
        row = self._run(lcdb_record=rec)
        # These columns come from char → catalog
        assert float(row["lcdb_period_hr"] or 0) == pytest.approx(8.879, rel=1e-3)
        assert int(row["lcdb_u_code"] or 0) == 3

    def test_lcdb_agreement_column_populated_when_record_provided(self):
        """When a found LCDB record is provided, lcdb_agreement must not be None/empty.

        We don't assert a specific agreement value here — that depends on which
        period the pipeline recovers from this particular synthetic LC, which is
        tested separately via compare_to_lcdb unit tests below.  The wiring test
        only cares that the column is populated (not 'no_prior' / None).
        """
        rec = _make_lcdb_record(period_hr=8.879, u_code=3)
        row = self._run(lcdb_record=rec)
        # Column must be populated with one of the known agreement strings
        assert row["lcdb_agreement"] in (
            "exact", "half_period", "double_period", "disagree"
        ), f"Expected agreement value, got: {row['lcdb_agreement']!r}"



# ── compare_to_lcdb unit tests (agreement category logic) ─────────────────────
# These test the agreement logic in isolation, without depending on period
# recovery from a noisy synthetic lightcurve.

class TestCompareToLcdb:
    """Unit tests for sources.lcdb.compare_to_lcdb — covers all four categories."""

    def _rec(self, period_hr, u_code=3, found=True):
        return _make_lcdb_record(period_hr=period_hr, u_code=u_code, found=found)

    def test_exact_agreement(self):
        from sources.lcdb import compare_to_lcdb
        rec = self._rec(8.879)
        result = compare_to_lcdb(8.879 * 1.02, rec)   # 2% off — within 5% tol
        assert result["agreement"] == "exact"
        assert result["within_tol"] is True

    def test_half_period(self):
        from sources.lcdb import compare_to_lcdb
        rec = self._rec(8.879)
        result = compare_to_lcdb(8.879 / 2.0 * 1.01, rec)
        assert result["agreement"] == "half_period"
        assert result["is_half"] is True

    def test_double_period(self):
        from sources.lcdb import compare_to_lcdb
        rec = self._rec(4.4)
        result = compare_to_lcdb(4.4 * 2.0 * 1.02, rec)
        assert result["agreement"] == "double_period"
        assert result["is_double"] is True

    def test_disagree(self):
        from sources.lcdb import compare_to_lcdb
        rec = self._rec(8.879)
        result = compare_to_lcdb(3.1, rec)    # unrelated period
        assert result["agreement"] == "disagree"

    def test_not_found_returns_no_prior(self):
        from sources.lcdb import compare_to_lcdb
        rec = self._rec(8.879, found=False)
        result = compare_to_lcdb(8.879, rec)
        assert result["agreement"] == "no_prior"

    def test_nan_period_returns_no_prior(self):
        from sources.lcdb import compare_to_lcdb
        rec = _make_lcdb_record(period_hr=float("nan"), found=True)
        result = compare_to_lcdb(8.879, rec)
        assert result["agreement"] == "no_prior"

    def test_delta_pct_is_fractional(self):
        from sources.lcdb import compare_to_lcdb
        rec = self._rec(10.0)
        result = compare_to_lcdb(11.0, rec)   # 10% off
        assert result["delta_pct"] == pytest.approx(0.10, rel=1e-3)

class TestWorkerLcdb:
    def test_worker_call_uses_lcdb_records_global(self):
        """_worker_call must pass lcdb_record from _worker_lcdb_records to run_single_asteroid."""
        import pipeline as pip

        provid  = "2025 MA19"
        rec     = _make_lcdb_record(provid)
        df_obj  = _make_synthetic_lc(provid=provid)

        pip._worker_df           = df_obj
        pip._worker_config       = PipelineConfig()
        pip._worker_config.tier.lrt_n_objects = 0
        pip._worker_lcdb_records = {provid: rec}

        calls = []
        original = pip.run_single_asteroid

        def spy(df, cfg, lcdb_record=None):
            calls.append(lcdb_record)
            return original(df, cfg, lcdb_record=lcdb_record)

        with patch("pipeline.run_single_asteroid", side_effect=spy), \
             patch("pipeline.load_single_object", return_value=df_obj):
            pip._worker_call(provid)

        assert len(calls) == 1
        assert calls[0] is rec

    def test_worker_call_none_when_not_in_records(self):
        """If provid not in _worker_lcdb_records, lcdb_record=None is passed."""
        import pipeline as pip

        provid = "2025 ZZ99"
        df_obj = _make_synthetic_lc(provid=provid)

        pip._worker_df           = df_obj
        pip._worker_config       = PipelineConfig()
        pip._worker_config.tier.lrt_n_objects = 0
        pip._worker_lcdb_records = {}   # empty — provid not present

        calls = []
        original = pip.run_single_asteroid

        def spy(df, cfg, lcdb_record=None):
            calls.append(lcdb_record)
            return original(df, cfg, lcdb_record=lcdb_record)

        with patch("pipeline.run_single_asteroid", side_effect=spy), \
             patch("pipeline.load_single_object", return_value=df_obj):
            pip._worker_call(provid)

        assert calls[0] is None

    def test_worker_call_handles_none_records_global(self):
        """If _worker_lcdb_records is None (not yet set), no crash."""
        import pipeline as pip

        provid = "2025 MA19"
        df_obj = _make_synthetic_lc(provid=provid)

        pip._worker_df           = df_obj
        pip._worker_config       = PipelineConfig()
        pip._worker_config.tier.lrt_n_objects = 0
        pip._worker_lcdb_records = None   # unset

        calls = []
        original = pip.run_single_asteroid

        def spy(df, cfg, lcdb_record=None):
            calls.append(lcdb_record)
            return original(df, cfg, lcdb_record=lcdb_record)

        with patch("pipeline.run_single_asteroid", side_effect=spy), \
             patch("pipeline.load_single_object", return_value=df_obj):
            pip._worker_call(provid)

        assert calls[0] is None
