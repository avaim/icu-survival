import numpy as np
import pandas as pd

from project1 import generate_feature_vector


def test_missing_static_value_becomes_nan():
    patient = pd.DataFrame(
        [
            ("00:00", "Age", 54.0),
            ("00:00", "Gender", 0.0),
            ("00:00", "Height", -1.0),
            ("00:00", "ICUType", 4.0),
            ("00:00", "Weight", 70.0),
        ],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_feature_vector(patient)

    assert np.isnan(features["Height"])


def test_timeseries_missing_observations_are_ignored_when_taking_max():
    patient = pd.DataFrame(
        [
            ("00:00", "Age", 54.0),
            ("00:00", "Gender", 0.0),
            ("00:00", "Height", 170.0),
            ("00:00", "ICUType", 4.0),
            ("00:00", "Weight", 70.0),
            ("00:07", "HR", -1.0),
            ("00:37", "HR", 77.0),
            ("01:37", "HR", 80.0),
        ],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_feature_vector(patient)

    assert features["max_HR"] == 80.0


def test_unlisted_variables_are_ignored():
    patient = pd.DataFrame(
        [
            ("00:00", "Age", 54.0),
            ("00:00", "Gender", 0.0),
            ("00:00", "Height", 170.0),
            ("00:00", "ICUType", 4.0),
            ("00:00", "Weight", 70.0),
            ("00:07", "MechVent", 1.0),
        ],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_feature_vector(patient)

    assert "max_MechVent" not in features


def test_variables_absent_from_patient_are_present_as_nan():
    patient = pd.DataFrame(
        [("00:00", "Age", 54.0)],
        columns=["Time", "Variable", "Value"],
    )

    features = generate_feature_vector(patient)

    assert set(features) == {
        "Age",
        "Gender",
        "Height",
        "ICUType",
        "Weight",
        "max_ALP",
        "max_ALT",
        "max_AST",
        "max_Albumin",
        "max_BUN",
        "max_Bilirubin",
        "max_Cholesterol",
        "max_Creatinine",
        "max_DiasABP",
        "max_FiO2",
        "max_GCS",
        "max_Glucose",
        "max_HCO3",
        "max_HCT",
        "max_HR",
        "max_K",
        "max_Lactate",
        "max_MAP",
        "max_Mg",
        "max_NIDiasABP",
        "max_NIMAP",
        "max_NISysABP",
        "max_Na",
        "max_PaCO2",
        "max_PaO2",
        "max_Platelets",
        "max_RespRate",
        "max_SaO2",
        "max_SysABP",
        "max_Temp",
        "max_TroponinI",
        "max_TroponinT",
        "max_Urine",
        "max_WBC",
        "max_pH",
    }
    assert np.isnan(features["Gender"])
    assert np.isnan(features["max_HR"])