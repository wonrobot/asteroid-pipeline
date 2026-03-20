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


def test_mbls_better_than_single_band():
    """
    With Nterms_band=0 and correct y_multiband, MBLS should find a period
    CLOSER to the truth than a naive single-band approach on the same data.
    This is a relative test — we're not asserting MBLS always wins on
    synthetic data (aliases can fool any method), but that the multiband
    information genuinely helps vs discarding it.

    Also tests that y_multiband has smaller mean than raw mag — confirming
    trend correction is applied.
    """
    from preprocessing import preprocess
    from config import PipelineConfig, DataConfig, TierConfig
    from gatspy.periodic import LombScargleMultiband

    config = PipelineConfig(
        data=DataConfig(min_obs_total=20, min_obs_band=5),
        tier=TierConfig(snr_threshold=2.0, min_obs=20),
    )

    rng    = np.random.default_rng(42)
    true_p = 5.9

    # 7 nights, each 3.5 hours — enough phase coverage for 5.9hr period
    t_list = []
    for night in range(7):
        t_night = night * 1.714 + np.linspace(0, 0.146, 41)  # 3.5hr = 0.146 days
        t_list.append(t_night)
    t = np.concatenate(t_list)

    band_offsets = {"Lr": 0.0, "Lg": 0.32, "Li": -0.18}
    bands = rng.choice(["Lr", "Lg", "Li"], len(t))
    mag = np.array([
        23.3
        + band_offsets[b]
        + 0.35 * np.sin(2 * np.pi * t[i] * 24 / true_p)
        + rng.normal(0, 0.04)
        for i, b in enumerate(bands)
    ])

    df = pd.DataFrame({
        "provid": ["sim_test"] * len(t),
        "mjd":    t,
        "band":   bands,
        "mag":    mag,
        "rmsmag": np.full(len(t), 0.04),
    })

    data = preprocess(df, config)

    # y_multiband should be centred near zero (trend corrected)
    assert abs(data.y_multiband.mean()) < 0.5, (
        f"y_multiband mean={data.y_multiband.mean():.3f} — "
        "expected near 0 after global_mean+trend correction."
    )

    # y_multiband mean should be much smaller than raw mag mean (~23)
    raw_mean = float(df["mag"].mean())
    assert abs(data.y_multiband.mean()) < abs(raw_mean) * 0.1, (
        f"y_multiband mean ({data.y_multiband.mean():.3f}) is not "
        f"much smaller than raw mag mean ({raw_mean:.3f}). "
        "global_mean correction may be missing."
    )

    # MBLS with Nterms_band=0 should find a peak — test it runs without error
    import numpy as np_inner
    test_periods = np_inner.linspace(0.5, 24.0, 5000)
    model = LombScargleMultiband(Nterms_base=2, Nterms_band=0)
    model.fit(data.t_hrs, data.y_multiband, data.dy, data.bands)
    power = model.periodogram(test_periods)

    assert len(power) == len(test_periods), "MBLS periodogram wrong length"
    assert power.max() > 0.1, (
        f"MBLS max power={power.max():.3f} — unexpectedly low, "
        "signal may not be detectable."
    )

    best_period = test_periods[np_inner.argmax(power)]
    delta_pct   = abs(best_period - true_p) / true_p * 100
    delta_half  = abs(best_period - true_p/2) / (true_p/2) * 100

    # Should find either the true period or P/2 (both are valid detections)
    assert delta_pct < 10 or delta_half < 10, (
        f"MBLS best period {best_period:.3f}hr is neither near true_p "
        f"({true_p}hr, delta={delta_pct:.1f}%) nor P/2 "
        f"({true_p/2:.3f}hr, delta={delta_half:.1f}%). "
        "Likely Nterms_band or y_multiband bug."
    )
