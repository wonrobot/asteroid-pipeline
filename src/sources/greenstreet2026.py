"""
greenstreet2026.py
------------------
Ground truth periods from Greenstreet et al. (2026), Tables 2 and 3.

Table 2: Primary (LSM) rotation periods used as ground truth in validation.
Table 3: Additional periods from high-order Fourier analysis.
         These are real physically-motivated periods for the same objects —
         not aliases. When the pipeline returns a period matching an additional
         period, it found a real solution, just not the primary one.

Usage
-----
    from sources.greenstreet2026 import GROUND_TRUTH, check_against_all_periods

    result = check_against_all_periods("2025 MK68", 5.554)
    # → {"match": "additional", "matched_period": 5.576, "relation": "exact"}
"""

import numpy as np

# Primary periods from Table 2 (LSM value, hours).
# superfast=True if P <= 2.2hr from at least one method.
GROUND_TRUTH = {
    "2025 MA19":  {"period_hr": 8.9,   "amplitude": 0.7,  "superfast": False},
    "2025 MA45":  {"period_hr": 1.6,   "amplitude": 0.7,  "superfast": True},
    "2025 MA46":  {"period_hr": 5.9,   "amplitude": 0.6,  "superfast": False},
    "2025 MC34":  {"period_hr": 8.4,   "amplitude": 0.8,  "superfast": False},
    "2025 MD38":  {"period_hr": 15.8,  "amplitude": 1.1,  "superfast": False},
    "2025 MD40":  {"period_hr": 4.4,   "amplitude": 0.7,  "superfast": False},
    "2025 MD67":  {"period_hr": 7.8,   "amplitude": 1.2,  "superfast": False},
    "2025 MD76":  {"period_hr": 11.0,  "amplitude": 0.7,  "superfast": False},
    "2025 ME15":  {"period_hr": 6.9,   "amplitude": 0.9,  "superfast": False},
    "2025 ME24":  {"period_hr": 2.9,   "amplitude": 0.3,  "superfast": False},
    "2025 ME68":  {"period_hr": 0.9,   "amplitude": 0.6,  "superfast": True},
    "2025 MF76":  {"period_hr": 2.2,   "amplitude": 0.2,  "superfast": True},
    "2025 MG17":  {"period_hr": 4.3,   "amplitude": 0.4,  "superfast": False},
    "2025 MG56":  {"period_hr": 0.3,   "amplitude": 0.5,  "superfast": True},
    "2025 MH40":  {"period_hr": 8.0,   "amplitude": 1.4,  "superfast": False},
    "2025 MH69":  {"period_hr": 6.7,   "amplitude": 0.7,  "superfast": False},
    "2025 MH75":  {"period_hr": 4.2,   "amplitude": 0.5,  "superfast": False},
    "2025 MH86":  {"period_hr": 4.4,   "amplitude": 0.5,  "superfast": False},
    "2025 MJ13":  {"period_hr": 3.4,   "amplitude": 0.6,  "superfast": False},
    "2025 MJ21":  {"period_hr": 3.4,   "amplitude": 0.3,  "superfast": False},
    "2025 MJ23":  {"period_hr": 7.4,   "amplitude": 0.8,  "superfast": False},
    "2025 MJ30":  {"period_hr": 5.6,   "amplitude": 0.5,  "superfast": False},
    "2025 MJ71":  {"period_hr": 0.031, "amplitude": 0.4,  "superfast": True},
    "2025 MJ79":  {"period_hr": 1.0,   "amplitude": 0.2,  "superfast": True},
    "2025 MK23":  {"period_hr": 6.2,   "amplitude": 0.9,  "superfast": False},
    "2025 MK41":  {"period_hr": 0.063, "amplitude": 0.2,  "superfast": True},
    "2025 MK68":  {"period_hr": 5.0,   "amplitude": 0.7,  "superfast": False},
    "2025 MK83":  {"period_hr": 6.1,   "amplitude": 0.5,  "superfast": False},
    "2025 MK88":  {"period_hr": 2.7,   "amplitude": 0.4,  "superfast": False},
    "2025 ML10":  {"period_hr": 7.0,   "amplitude": 1.0,  "superfast": False},
    "2025 ML17":  {"period_hr": 6.7,   "amplitude": 0.6,  "superfast": False},
    "2025 ML35":  {"period_hr": 21.3,  "amplitude": 0.8,  "superfast": False},
    "2025 ML52":  {"period_hr": 11.5,  "amplitude": 0.7,  "superfast": False},
    "2025 ML53":  {"period_hr": 5.2,   "amplitude": 0.7,  "superfast": False},
    "2025 MM37":  {"period_hr": 3.7,   "amplitude": 0.3,  "superfast": True},
    "2025 MM81":  {"period_hr": 1.1,   "amplitude": 1.0,  "superfast": True},
    "2025 MM82":  {"period_hr": 5.0,   "amplitude": 0.8,  "superfast": False},
    "2025 MN25":  {"period_hr": 0.4,   "amplitude": 0.4,  "superfast": True},
    "2025 MN37":  {"period_hr": 4.8,   "amplitude": 0.8,  "superfast": False},
    "2025 MN45":  {"period_hr": 0.031, "amplitude": 0.4,  "superfast": True},
    "2025 MN7":   {"period_hr": 6.8,   "amplitude": 0.7,  "superfast": False},
    "2025 MO35":  {"period_hr": 6.3,   "amplitude": 0.5,  "superfast": False},
    "2025 MO39":  {"period_hr": 4.9,   "amplitude": 0.9,  "superfast": False},
    "2025 MO47":  {"period_hr": 9.1,   "amplitude": 0.7,  "superfast": False},
    "2025 MO79":  {"period_hr": 5.5,   "amplitude": 0.6,  "superfast": False},
    "2025 MP21":  {"period_hr": 6.2,   "amplitude": 0.6,  "superfast": False},
    "2025 MP47":  {"period_hr": 4.9,   "amplitude": 0.4,  "superfast": True},
    "2025 MP61":  {"period_hr": 3.0,   "amplitude": 0.6,  "superfast": False},
    "2025 MP67":  {"period_hr": 4.1,   "amplitude": 0.7,  "superfast": False},
    "2025 MP71":  {"period_hr": 9.1,   "amplitude": 0.4,  "superfast": False},
    "2025 MQ58":  {"period_hr": 2.9,   "amplitude": 0.3,  "superfast": False},
    "2025 MR33":  {"period_hr": 3.5,   "amplitude": 0.3,  "superfast": False},
    "2025 MS34":  {"period_hr": 2.3,   "amplitude": 0.5,  "superfast": False},
    "2025 MS7":   {"period_hr": 4.6,   "amplitude": 0.8,  "superfast": False},
    "2025 MT24":  {"period_hr": 8.9,   "amplitude": 1.0,  "superfast": False},
    "2025 MU10":  {"period_hr": 6.5,   "amplitude": 0.5,  "superfast": False},
    "2025 MU15":  {"period_hr": 0.4,   "amplitude": 0.5,  "superfast": True},
    "2025 MU24":  {"period_hr": 2.2,   "amplitude": 0.3,  "superfast": True},
    "2025 MU59":  {"period_hr": 8.2,   "amplitude": 0.6,  "superfast": False},
    "2025 MU8":   {"period_hr": 0.8,   "amplitude": 0.6,  "superfast": True},
    "2025 MU9":   {"period_hr": 4.9,   "amplitude": 0.6,  "superfast": False},
    "2025 MV19":  {"period_hr": 7.4,   "amplitude": 0.7,  "superfast": False},
    "2025 MV31":  {"period_hr": 5.2,   "amplitude": 0.7,  "superfast": False},
    "2025 MV38":  {"period_hr": 6.0,   "amplitude": 0.6,  "superfast": False},
    "2025 MV4":   {"period_hr": 5.9,   "amplitude": 0.8,  "superfast": False},
    "2025 MV46":  {"period_hr": 3.4,   "amplitude": 0.2,  "superfast": False},
    "2025 MV71":  {"period_hr": 0.2,   "amplitude": 0.4,  "superfast": True},
    "2025 MW70":  {"period_hr": 3.9,   "amplitude": 0.2,  "superfast": False},
    "2025 MX34":  {"period_hr": 5.8,   "amplitude": 0.7,  "superfast": False},
    "2025 MX44":  {"period_hr": 1.1,   "amplitude": 0.2,  "superfast": True},
    "2025 MX50":  {"period_hr": 1.9,   "amplitude": 0.3,  "superfast": True},
    "2025 MX63":  {"period_hr": 8.3,   "amplitude": 0.7,  "superfast": False},
    "2025 MX69":  {"period_hr": 9.1,   "amplitude": 0.7,  "superfast": False},
    "2025 MY23":  {"period_hr": 3.1,   "amplitude": 0.2,  "superfast": False},
    "2025 MY77":  {"period_hr": 7.6,   "amplitude": 1.0,  "superfast": False},
    "2025 MZ78":  {"period_hr": 1.2,   "amplitude": 0.5,  "superfast": True},
}

# Additional periods from Table 3 (high-order Fourier analysis).
# These are real physically-motivated periods, not pure aliases.
# Key: provid. Value: list of additional period values in hours.
ADDITIONAL_PERIODS = {
    "2025 MA19":  [10.913, 14.201],
    "2025 MA45":  [0.817],
    "2025 MA46":  [5.225, 6.685],
    "2025 MC34":  [5.075, 9.188, 10.144, 15.216],
    "2025 MD38":  [7.906, 11.856, 23.696, 47.392],
    "2025 MD40":  [4.006, 4.811],
    "2025 MD67":  [3.311, 5.879],
    "2025 ME15":  [4.849, 6.041, 6.902],
    "2025 MF76":  [1.865, 1.869, 2.217],
    "2025 MH40":  [8.023, 12.028, 48.200],
    "2025 MH86":  [4.019, 4.835],
    "2025 MJ30":  [5.009],
    "2025 MK23":  [3.090, 6.180, 9.273, 14.191, 17.732, 20.793],
    "2025 MK68":  [4.548, 5.576, 5.604, 6.266, 8.357],
    "2025 MK83":  [5.391, 5.398],
    "2025 ML10":  [4.918, 6.151, 8.196],
    "2025 ML17":  [5.253, 5.896, 7.827, 7.841],
    "2025 ML35":  [14.711, 21.662, 38.255, 38.424],
    "2025 ML52":  [11.439, 13.842, 17.159],
    "2025 ML53":  [6.104],
    "2025 MM37":  [1.720, 1.724, 2.006],
    "2025 MM82":  [5.609],
    "2025 MN37":  [9.034],
    "2025 MO35":  [5.551],
    "2025 MO39":  [4.952, 7.056, 7.067, 8.229],
    "2025 MO79":  [3.094, 3.555, 4.912, 4.915, 5.481, 6.184],
    "2025 MP21":  [6.220],
    "2025 MP47":  [4.036, 4.409, 4.858, 4.861, 4.868],
    "2025 MP67":  [3.540, 5.499, 5.520],
    "2025 MS34":  [2.209, 2.432],
    "2025 MS7":   [3.832, 4.165, 5.042],
    "2025 MT24":  [6.673],
    "2025 MU59":  [4.093],
    "2025 MU8":   [0.838],
    "2025 MV31":  [5.168, 5.177, 5.190, 5.832, 6.497],
    "2025 MV38":  [4.797, 5.332, 6.858, 6.879],
    "2025 MV4":   [4.742, 5.917, 5.925, 7.900],
    "2025 MW70":  [3.336, 3.591, 4.205, 4.633, 4.643],
    "2025 MX44":  [1.098],
    "2025 MX50":  [3.882],
    "2025 MX69":  [5.769, 7.603, 9.109],
    "2025 MY77":  [],  # not in Table 3
}


def check_against_all_periods(
    provid: str,
    pipe_period: float,
    tolerance: float = 0.10,
) -> dict:
    """
    Check a pipeline period against Greenstreet primary AND additional periods
    and their harmonics (P/2, 2P, P/3, 3P).

    Returns
    -------
    dict with keys:
        match        : "primary" | "additional" | "harmonic_of_primary" |
                       "harmonic_of_additional" | "no_match"
        matched_period: float — which Greenstreet period it matched (NaN if none)
        relation     : "exact" | "P/2" | "2P" | "P/3" | "3P" | None
        delta_pct    : fractional difference from the matched period (%)
    """
    import numpy as np

    gt = GROUND_TRUTH.get(provid)
    if gt is None or np.isnan(pipe_period):
        return {"match": "no_truth", "matched_period": np.nan,
                "relation": None, "delta_pct": np.nan}

    primary    = gt["period_hr"]
    additional = ADDITIONAL_PERIODS.get(provid, [])

    harmonics = [(1.0, "exact"), (0.5, "P/2"), (2.0, "2P"),
                 (1/3, "P/3"), (3.0, "3P")]

    # Check primary first
    for mult, label in harmonics:
        ref = primary * mult
        delta = abs(pipe_period - ref) / ref
        if delta <= tolerance:
            rel = "exact" if label == "exact" else label
            kind = "primary" if label == "exact" else "harmonic_of_primary"
            return {"match": kind, "matched_period": primary,
                    "relation": rel, "delta_pct": round(delta * 100, 1)}

    # Check additional periods
    for add_p in additional:
        for mult, label in harmonics:
            ref = add_p * mult
            delta = abs(pipe_period - ref) / ref
            if delta <= tolerance:
                rel = "exact" if label == "exact" else label
                kind = "additional" if label == "exact" else "harmonic_of_additional"
                return {"match": kind, "matched_period": add_p,
                        "relation": rel, "delta_pct": round(delta * 100, 1)}

    return {"match": "no_match", "matched_period": np.nan,
            "relation": None, "delta_pct": np.nan}
