"""
test_mbls_regression.py
-----------------------
Regression tests for the two MBLS fixes that were silently lost between
Colab sessions. These tests catch the bugs before they affect science output.

Bug 1: Nterms_band=1 let each band fit its own sinusoid, defeating the
       joint period constraint. Fix: Nterms_band=0.

Bug 2: y_multiband was raw ~23 mag with no trend removal. MBLS lost
       numerical precision on the ~0.1 mag periodic signal.
       Fix: subtract global_mean + polynomial trend.
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_nterms_band_is_zero():
    """MBLS must use Nterms_band=0 — shared lightcurve shape across bands."""
    path = os.path.join(os.path.dirname(__file__), "..", "src", "tier1.py")
    with open(path) as f:
        content = f.read()
    assert "Nterms_band=0" in content, (
        "tier1.py uses Nterms_band=1. This lets each band fit its own "
        "sinusoid, defeating the joint constraint. Must be 0."
    )
    assert "Nterms_band=1" not in content, (
        "tier1.py still contains Nterms_band=1."
    )


def test_y_multiband_global_mean_present():
    """preprocessing.py must subtract global_mean from y_multiband."""
    path = os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing.py")
    with open(path) as f:
        content = f.read()
    assert "global_mean" in content, (
        "preprocessing.py missing global_mean subtraction from y_multiband. "
        "Without this MBLS receives raw ~23 mag values."
    )


def test_two_data_paths_differ():
    """y_dt and y_multiband must be different arrays after preprocessing."""
    from preprocessing import preprocess
    from config import PipelineConfig

    config = PipelineConfig()
    rng = np.random.default_rng(42)
    n   = 120

    # Simulate 3-band data with realistic colour offsets and a 5.9hr period
    t = np.sort(rng.uniform(0, 12, n))
    band_offsets = {"Lr": 0.0, "Lg": 0.3, "Li": -0.15}
    bands = rng.choice(["Lr", "Lg", "Li"], n)
    mag = np.array([
        23.0
        + band_offsets[b]
        + 0.3 * np.sin(2 * np.pi * t[i] / 5.9)
        + rng.normal(0, 0.05)
        for i, b in enumerate(bands)
    ])

    df = pd.DataFrame({
        "provid":  ["test"] * n,
        "mjd":     t,
        "band":    bands,
        "mag":     mag,
        "rmsmag":  np.full(n, 0.05),
    })

    data = preprocess(df, config)

    assert not np.allclose(data.y_dt, data.y_multiband), (
        "y_dt and y_multiband are identical — two data paths not maintained."
    )
    assert abs(data.y_multiband.mean()) < 0.5, (
        f"y_multiband mean={data.y_multiband.mean():.3f} — expected near 0."
    )
    assert abs(data.y_dt.mean()) < 0.01, (
        f"y_dt mean={data.y_dt.mean():.4f} — expected near 0."
    )


def test_mbls_period_ma46_equivalent():
    """
    On simulated MA46-like data (P=5.9hr, 3 bands, dense),
    MBLS should find a period within 5% of truth.
    This catches Nterms_band and y_multiband bugs simultaneously.
    """
    from preprocessing import preprocess
    from tier1 import run_tier1
    from tier2 import run_tier2
    from config import PipelineConfig, DataConfig, TierConfig

    config = PipelineConfig(
        data=DataConfig(min_obs_total=20, min_obs_band=5),
        tier=TierConfig(snr_threshold=2.0, min_obs=20),
    )

    rng      = np.random.default_rng(99)
    true_p   = 5.9
    n        = 289
    baseline = 12.0

    # Simulate 7 nights of dense observations
    t_list = []
    for night in range(7):
        t_night = night * (baseline / 7) + rng.uniform(0, 0.1, 41)
        t_list.append(np.sort(t_night))
    t = np.concatenate(t_list)

    band_offsets = {"Lr": 0.0, "Lg": 0.32, "Li": -0.18}
    bands = rng.choice(["Lr", "Lg", "Li"], len(t))
    mag = np.array([
        23.3
        + band_offsets[b]
        + 0.28 * np.sin(2 * np.pi * t[i] * 24 / true_p)
        + rng.normal(0, 0.05)
        for i, b in enumerate(bands)
    ])

    df = pd.DataFrame({
        "provid": ["sim_ma46"] * len(t),
        "mjd":    t,
        "band":   bands,
        "mag":    mag,
        "rmsmag": np.full(len(t), 0.05),
    })

    data = preprocess(df, config)
    t1   = run_tier1(data, config)

    assert t1.passes, f"Tier1 failed on simulated data: {t1.reject_reason}"

    t2 = run_tier2(data, t1, config)

    mbls_delta = abs(t2.best_period_mbls - true_p) / true_p
    assert mbls_delta < 0.05, (
        f"MBLS period {t2.best_period_mbls:.3f}hr is >5% from "
        f"true {true_p}hr (delta={mbls_delta*100:.1f}%). "
        f"Likely Nterms_band or y_multiband bug."
    )
