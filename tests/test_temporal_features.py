import numpy as np
import pandas as pd
import pytest

from temporal_features import generate_temporal_features


EXPECTED_FEATURE_NAMES = [
    f"{variable}_{suffix}"
    for variable in ("HR", "RespRate", "Temp", "MAP", "GCS")
    for suffix in (
        "early_mean",
        "late_mean",
        "delta_mean",
        "early_missing",
        "late_missing",
    )
]


def test_temporal_features_have_fixed_count_and_order():
    features = generate_temporal_features(
        pd.DataFrame(columns=["Time", "Variable", "Value"])
    )

    assert len(features) == 25
    assert list(features) == EXPECTED_FEATURE_NAMES
    assert all(
        np.isnan(features[name])
        for name in EXPECTED_FEATURE_NAMES
        if name.endswith(("_mean", "_delta_mean"))
    )
    assert all(
        features[name] == 1.0
        for name in EXPECTED_FEATURE_NAMES
        if name.endswith("_missing")
    )


def test_windows_include_expected_boundaries_and_compute_delta():
    df = pd.DataFrame(
        [
            ("23:59", "HR", 10.0),
            ("24:00", "HR", 20.0),
            ("48:00", "HR", 40.0),
            ("48:01", "HR", 100.0),
        ],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_temporal_features(df)

    assert features["HR_early_mean"] == 10.0
    assert features["HR_late_mean"] == 30.0
    assert features["HR_delta_mean"] == 20.0
    assert features["HR_early_missing"] == 0.0
    assert features["HR_late_missing"] == 0.0


def test_missing_sentinel_is_excluded_but_true_zero_is_valid():
    df = pd.DataFrame(
        [
            ("01:00", "RespRate", 0.0),
            ("02:00", "RespRate", -1.0),
            ("25:00", "RespRate", -1.0),
        ],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_temporal_features(df)

    assert features["RespRate_early_mean"] == 0.0
    assert features["RespRate_early_missing"] == 0.0
    assert np.isnan(features["RespRate_late_mean"])
    assert np.isnan(features["RespRate_delta_mean"])
    assert features["RespRate_late_missing"] == 1.0


def test_absent_variable_has_missing_windows():
    df = pd.DataFrame(
        [("00:00", "HR", 70.0)],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_temporal_features(df)

    assert np.isnan(features["MAP_early_mean"])
    assert np.isnan(features["MAP_late_mean"])
    assert np.isnan(features["MAP_delta_mean"])
    assert features["MAP_early_missing"] == 1.0
    assert features["MAP_late_missing"] == 1.0


def test_manually_calculated_means_and_deltas():
    df = pd.DataFrame(
        [
            ("00:00", "HR", 60.0),
            ("12:00", "HR", 80.0),
            ("24:00", "HR", 100.0),
            ("48:00", "HR", 120.0),
            ("01:00", "Temp", 36.0),
            ("23:00", "Temp", 38.0),
            ("25:00", "Temp", 37.0),
            ("47:00", "Temp", 39.0),
        ],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_temporal_features(df)

    assert features["HR_early_mean"] == 70.0
    assert features["HR_late_mean"] == 110.0
    assert features["HR_delta_mean"] == 40.0
    assert features["Temp_early_mean"] == 37.0
    assert features["Temp_late_mean"] == 38.0
    assert features["Temp_delta_mean"] == 1.0


@pytest.mark.parametrize("missing_column", ["Time", "Variable", "Value"])
def test_missing_required_input_column_raises_key_error(missing_column):
    df = pd.DataFrame(
        [("00:00", "HR", 70.0)],
        columns=["Time", "Variable", "Value"],
    ).drop(columns=[missing_column])

    with pytest.raises(KeyError):
        generate_temporal_features(df)


def test_incorrect_input_column_name_raises_key_error():
    df = pd.DataFrame(
        [("00:00", "HR", 70.0)],
        columns=["Time", "Variable", "Measurement"],
    )

    with pytest.raises(KeyError):
        generate_temporal_features(df)