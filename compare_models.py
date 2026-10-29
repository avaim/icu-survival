"""Compare baseline ICU features with added temporal features."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project1 import generate_feature_vector
from temporal_features import generate_temporal_features


N_PATIENTS = 2_000
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 3
MODEL_RANDOM_STATE = 42
CV_FOLDS = 5
CV_RANDOM_STATE = 42
TARGET_COLUMN = "In-hospital_death"
DATA_DIRECTORY = Path("data/files")


def load_feature_datasets() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load labels and construct aligned baseline and temporal feature arrays."""
    labels = pd.read_csv("data/labels.csv").iloc[:N_PATIENTS].reset_index(drop=True)
    y = labels[TARGET_COLUMN].to_numpy(dtype=int)

    baseline_rows = []
    enhanced_rows = []
    baseline_feature_names = None
    enhanced_feature_names = None

    for record_id in labels["RecordID"]:
        patient_df = pd.read_csv(DATA_DIRECTORY / f"{int(record_id)}.csv")
        baseline_features = generate_feature_vector(patient_df)
        temporal_features = generate_temporal_features(patient_df)

        if baseline_feature_names is None:
            baseline_feature_names = list(baseline_features)
            enhanced_feature_names = baseline_feature_names + list(temporal_features)

        baseline_rows.append([baseline_features[name] for name in baseline_feature_names])
        enhanced_rows.append(
            [baseline_features[name] for name in baseline_feature_names]
            + [temporal_features[name] for name in temporal_features]
        )

    return (
        np.asarray(baseline_rows, dtype=float),
        np.asarray(enhanced_rows, dtype=float),
        y,
        baseline_feature_names,
        enhanced_feature_names,
    )


def make_model_pipeline() -> Pipeline:
    """Create the shared training-only preprocessing and logistic model."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    fit_intercept=False,
                    random_state=MODEL_RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute classification and ranking metrics on one held-out split."""
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[-1, 1]).ravel()
    sensitivity_denominator = tp + fn
    specificity_denominator = tn + fp

    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "F1-score": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "AUROC": float(roc_auc_score(y_test, scores)),
        "Average precision": float(average_precision_score(y_test, scores)),
        "Sensitivity": float(tp / sensitivity_denominator) if sensitivity_denominator else 0.0,
        "Specificity": float(tn / specificity_denominator) if specificity_denominator else 0.0,
    }


def cross_validate_models(
    X_baseline: np.ndarray,
    X_enhanced: np.ndarray,
    y: np.ndarray,
) -> dict[str, tuple[float, float, float, float]]:
    """Evaluate both feature sets over identical stratified folds.

    Each returned metric maps to baseline mean/std followed by enhanced
    mean/std. A population standard deviation (ddof=0) summarizes the five
    fold scores.
    """
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    folds = list(splitter.split(X_baseline, y))
    baseline_fold_scores: dict[str, list[float]] = {}
    enhanced_fold_scores: dict[str, list[float]] = {}

    for train_indices, validation_indices in folds:
        y_train = y[train_indices]
        y_validation = y[validation_indices]

        baseline_model = make_model_pipeline()
        enhanced_model = make_model_pipeline()
        baseline_model.fit(X_baseline[train_indices], y_train)
        enhanced_model.fit(X_enhanced[train_indices], y_train)

        baseline_scores = evaluate_model(
            baseline_model, X_baseline[validation_indices], y_validation
        )
        enhanced_scores = evaluate_model(
            enhanced_model, X_enhanced[validation_indices], y_validation
        )

        for metric, score in baseline_scores.items():
            baseline_fold_scores.setdefault(metric, []).append(score)
        for metric, score in enhanced_scores.items():
            enhanced_fold_scores.setdefault(metric, []).append(score)

    results = {}
    for metric, baseline_scores in baseline_fold_scores.items():
        enhanced_scores = enhanced_fold_scores[metric]
        results[metric] = (
            float(np.mean(baseline_scores)),
            float(np.std(baseline_scores, ddof=0)),
            float(np.mean(enhanced_scores)),
            float(np.std(enhanced_scores, ddof=0)),
        )
    return results


def print_cross_validation_results(
    results: dict[str, tuple[float, float, float, float]],
    baseline_feature_count: int,
    enhanced_feature_count: int,
) -> None:
    """Print means, standard deviations, and direction of CV differences."""
    print(f"\n{CV_FOLDS}-Fold Stratified Cross-Validation")
    print(f"Fold random_state={CV_RANDOM_STATE}; standard deviation uses ddof=0")
    print(
        f"Feature count: baseline={baseline_feature_count}, "
        f"enhanced={enhanced_feature_count}"
    )
    print(
        f"{'Metric':<22} {'Baseline mean (SD)':>19} "
        f"{'Enhanced mean (SD)':>19} {'Change':>12}"
    )
    print("-" * 77)
    for metric, (baseline_mean, baseline_std, enhanced_mean, enhanced_std) in results.items():
        difference = enhanced_mean - baseline_mean
        if np.isclose(difference, 0.0):
            change = "No change"
        elif difference > 0:
            change = "Improved"
        else:
            change = "Declined"
        print(
            f"{metric:<22} "
            f"{baseline_mean:.4f} ({baseline_std:.4f}) "
            f"{enhanced_mean:.4f} ({enhanced_std:.4f}) "
            f"{change:>12}"
        )


def main() -> None:
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    np.random.seed(config["seed"])

    X_baseline, X_enhanced, y, baseline_names, enhanced_names = load_feature_datasets()
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y,
    )

    X_train_baseline, X_test_baseline = X_baseline[train_indices], X_baseline[test_indices]
    X_train_enhanced, X_test_enhanced = X_enhanced[train_indices], X_enhanced[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    baseline_model = make_model_pipeline()
    enhanced_model = make_model_pipeline()
    baseline_model.fit(X_train_baseline, y_train)
    enhanced_model.fit(X_train_enhanced, y_train)

    baseline_metrics = evaluate_model(baseline_model, X_test_baseline, y_test)
    enhanced_metrics = evaluate_model(enhanced_model, X_test_enhanced, y_test)

    print("ICU Survival Model Comparison")
    print(f"Patients: {len(y)} | Train: {len(train_indices)} | Test: {len(test_indices)}")
    print(f"Split: stratified 80/20, random_state={SPLIT_RANDOM_STATE}")
    print(f"Logistic regression: L2, C=1.0, solver=liblinear, random_state={MODEL_RANDOM_STATE}")
    print(f"Feature count: baseline={len(baseline_names)}, enhanced={len(enhanced_names)}")
    print()
    print(f"{'Metric':<22} {'Baseline':>12} {'Enhanced':>12} {'Difference':>12}")
    print("-" * 60)
    for metric in baseline_metrics:
        baseline_score = baseline_metrics[metric]
        enhanced_score = enhanced_metrics[metric]
        difference = enhanced_score - baseline_score
        print(f"{metric:<22} {baseline_score:>12.4f} {enhanced_score:>12.4f} {difference:>+12.4f}")

    cross_validation_results = cross_validate_models(X_baseline, X_enhanced, y)
    print_cross_validation_results(
        cross_validation_results,
        len(baseline_names),
        len(enhanced_names),
    )


if __name__ == "__main__":
    main()
