from pathlib import Path

import pandas as pd

from temporal_features import generate_temporal_features


PATIENT_ID_TO_INSPECT = 132539
VITAL_SIGNS = ("HR", "RespRate", "Temp", "MAP", "GCS")
ADDITIONAL_VITAL_SIGNS = ("MAP", "RespRate", "Temp", "GCS")
FEATURE_SUFFIXES = (
    "early_mean",
    "late_mean",
    "delta_mean",
    "early_missing",
    "late_missing",
)


def _elapsed_hours(time_values: pd.Series) -> pd.Series:
    """Convert elapsed HH:MM strings into numeric hours."""
    parts = time_values.astype(str).str.partition(":")
    hours = pd.to_numeric(parts[0], errors="raise")
    minutes = pd.to_numeric(parts[2], errors="raise")
    return hours + minutes / 60.0


def _valid_window_values(
    patient_df: pd.DataFrame,
    variable: str,
    early_window: bool,
) -> pd.Series:
    """Return non-missing values for a vital sign in one requested window."""
    elapsed = _elapsed_hours(patient_df["Time"])
    values = pd.to_numeric(patient_df["Value"], errors="coerce")
    in_window = (
        (elapsed >= 0) & (elapsed < 24)
        if early_window
        else (elapsed >= 24) & (elapsed <= 48)
    )
    matching = patient_df["Variable"].eq(variable) & in_window
    return values[matching & values.notna() & values.ne(-1)]


def _affected_vitals(patient_df: pd.DataFrame) -> list[str]:
    """Find vital signs without valid readings in either temporal window."""
    affected = []
    for variable in VITAL_SIGNS:
        early_values = _valid_window_values(patient_df, variable, early_window=True)
        late_values = _valid_window_values(patient_df, variable, early_window=False)
        if early_values.empty or late_values.empty:
            affected.append(variable)
    return affected


def _display_vital(
    patient_df: pd.DataFrame,
    variable: str,
    features: dict[str, float],
) -> None:
    """Print raw window observations and generated features for one vital."""
    print(f"\n  Vital sign: {variable}")
    elapsed = _elapsed_hours(patient_df["Time"])
    measurements = patient_df.loc[
        patient_df["Variable"].eq(variable), ["Time", "Value"]
    ].copy()
    measurements["ElapsedHour"] = elapsed.loc[measurements.index]

    windows = (
        ("Early window (0 <= hour < 24)", (measurements["ElapsedHour"] >= 0) & (measurements["ElapsedHour"] < 24)),
        ("Late window (24 <= hour <= 48)", (measurements["ElapsedHour"] >= 24) & (measurements["ElapsedHour"] <= 48)),
    )
    for label, mask in windows:
        print(f"    Raw measurements, {label}:")
        window_rows = measurements.loc[mask, ["Time", "Value"]]
        if window_rows.empty:
            print("      (no rows)")
        else:
            print(window_rows.to_string(index=False, header=True).replace("\n", "\n      "))

    print("    Generated temporal features:")
    for suffix in FEATURE_SUFFIXES:
        feature_name = f"{variable}_{suffix}"
        print(f"      {feature_name}: {features[feature_name]}")


def _load_patient(patient_id: int) -> pd.DataFrame:
    return pd.read_csv(Path("data/files") / f"{patient_id}.csv")


def _find_missing_data_patients(exclude_patient_id: int, count: int = 3):
    """Find source records with a vital missing valid data in either window."""
    matches = []
    for path in sorted(Path("data/files").glob("*.csv")):
        patient_id = int(path.stem)
        if patient_id == exclude_patient_id:
            continue

        patient_df = pd.read_csv(path)
        affected = _affected_vitals(patient_df)
        if affected:
            matches.append((patient_id, patient_df, affected))
        if len(matches) == count:
            break
    return matches


def main() -> None:
    patient_df = _load_patient(PATIENT_ID_TO_INSPECT)
    patient_features = generate_temporal_features(patient_df)

    print(f"\n{'=' * 78}")
    print(f"Patient ID: {PATIENT_ID_TO_INSPECT} (additional vital signs)")
    print(f"{'=' * 78}")
    for variable in ADDITIONAL_VITAL_SIGNS:
        _display_vital(patient_df, variable, patient_features)

    missing_patients = _find_missing_data_patients(PATIENT_ID_TO_INSPECT)
    print(f"\n{'=' * 78}")
    print("Patients with a vital sign missing valid observations in a window")
    print(f"{'=' * 78}")

    if not missing_patients:
        print("No matching patient records were found.")
        return

    for patient_id, patient_df, affected in missing_patients:
        print(f"\nPatient ID: {patient_id}")
        print(f"Affected vital signs: {', '.join(affected)}")
        features = generate_temporal_features(patient_df)
        for variable in affected:
            _display_vital(patient_df, variable, features)


if __name__ == "__main__":
    main()
