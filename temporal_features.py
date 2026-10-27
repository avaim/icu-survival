"""Temporal feature extraction for the ICU survival project."""

import numpy as np
import pandas as pd


TEMPORAL_VARIABLES = ("HR", "RespRate", "Temp", "MAP", "GCS")
FEATURE_SUFFIXES = (
    "early_mean",
    "late_mean",
    "delta_mean",
    "early_missing",
    "late_missing",
)


def generate_temporal_features(df: pd.DataFrame) -> dict[str, float]:
    """Generate ordered early/late summary features for five ICU variables.

    ``df`` must contain ``Time``, ``Variable``, and ``Value`` columns. Times
    are elapsed ICU time formatted as ``HH:MM``; hours may exceed 23. The
    early window is ``0 <= hour < 24`` and the late window is
    ``24 <= hour <= 48``.

    A value of ``-1`` is treated as a missing observation and excluded from
    means. A window's missing indicator is 1 when it has no valid
    observations (so its mean is unavailable), otherwise 0. This interprets
    missingness as an unavailable window summary; it does not infer expected
    measurement times from the irregular observations. True zero values are
    valid observations.

    Returns:
        An insertion-ordered dictionary of exactly 25 features, grouped by
        variable in ``HR``, ``RespRate``, ``Temp``, ``MAP``, ``GCS`` order.
        Means and deltas are ``numpy.nan`` when either required window has no
        valid observations. The delta is late mean minus early mean.
    """
    if df.empty:
        elapsed_hours = pd.Series(index=df.index, dtype=float)
    else:
        time_parts = df["Time"].astype(str).str.partition(":")
        hours = pd.to_numeric(time_parts[0], errors="raise")
        minutes = pd.to_numeric(time_parts[2], errors="raise")
        elapsed_hours = hours + minutes / 60.0

    values = pd.to_numeric(df["Value"], errors="coerce").replace(-1, np.nan)
    result: dict[str, float] = {}

    for variable in TEMPORAL_VARIABLES:
        matching = df["Variable"] == variable
        early = values[matching & (elapsed_hours >= 0) & (elapsed_hours < 24)].dropna()
        late = values[matching & (elapsed_hours >= 24) & (elapsed_hours <= 48)].dropna()

        early_mean = float(early.mean()) if not early.empty else np.nan
        late_mean = float(late.mean()) if not late.empty else np.nan
        delta_mean = late_mean - early_mean if not np.isnan(early_mean) and not np.isnan(late_mean) else np.nan

        result[f"{variable}_early_mean"] = early_mean
        result[f"{variable}_late_mean"] = late_mean
        result[f"{variable}_delta_mean"] = delta_mean
        result[f"{variable}_early_missing"] = float(early.empty)
        result[f"{variable}_late_missing"] = float(late.empty)

    return result
