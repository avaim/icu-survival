import numpy as np
import pytest
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression

from project1 import cv_performance, performance


class FixedScoreClassifier:
    def fit(self, X, y):
        return self

    def decision_function(self, X):
        return X[:, 0]

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("accuracy", 2 / 3),
        ("precision", 2 / 3),
        ("f1_score", 2 / 3),
        ("auroc", 7 / 9),
        ("average_precision", 29 / 36),
        ("sensitivity", 2 / 3),
        ("specificity", 2 / 3),
    ],
)
def test_performance_returns_requested_metric(metric, expected):
    y = np.array([1, 1, 1, -1, -1, -1])
    X = np.array([[0.8], [0.2], [-0.1], [0.5], [-0.4], [-0.2]])

    result = performance(FixedScoreClassifier(), X, y, metric=metric)

    assert result == pytest.approx(expected)


def test_performance_handles_no_positive_predictions():
    y = np.array([1, 1, -1, -1])
    X = np.array([[-0.2], [-0.1], [-0.4], [-0.3]])
    classifier = FixedScoreClassifier()

    assert performance(classifier, X, y, "precision") == 0.0
    assert performance(classifier, X, y, "f1_score") == 0.0
    assert performance(classifier, X, y, "sensitivity") == 0.0
    assert performance(classifier, X, y, "specificity") == 1.0


def test_performance_handles_missing_actual_classes():
    classifier = FixedScoreClassifier()
    all_negative_y = np.array([-1, -1])
    all_negative_X = np.array([[-0.2], [-0.1]])
    all_positive_y = np.array([1, 1])
    all_positive_X = np.array([[0.2], [0.1]])

    assert performance(classifier, all_negative_X, all_negative_y, "sensitivity") == 0.0
    assert performance(classifier, all_negative_X, all_negative_y, "specificity") == 1.0
    assert performance(classifier, all_positive_X, all_positive_y, "sensitivity") == 1.0
    assert performance(classifier, all_positive_X, all_positive_y, "specificity") == 0.0


def test_specificity_and_zero_negative_denominator():
    classifier = FixedScoreClassifier()
    y = np.array([-1, -1, 1, 1])
    X = np.array([[-0.5], [0.4], [0.2], [-0.1]])
    all_positive_y = np.array([1, 1])
    all_positive_X = np.array([[0.2], [0.4]])

    assert performance(classifier, X, y, "specificity") == 0.5
    assert performance(classifier, all_positive_X, all_positive_y, "specificity") == 0.0


def test_performance_handles_logistic_regression_scores():
    X = np.array([[-4.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    classifier = LogisticRegression().fit(X, y)

    assert performance(classifier, X, y, "accuracy") == 1.0


def test_performance_handles_kernel_ridge_predictions_as_scores():
    X = np.array([[-4.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    classifier = KernelRidge(alpha=0.01, kernel="linear").fit(X, y)

    assert performance(classifier, X, y, "accuracy") == 1.0


@pytest.mark.parametrize(
    "metric",
    [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ],
)
def test_cv_performance_uses_shared_metric_and_keeps_summary_tuple(metric):
    X = np.array([[-0.8], [-0.4], [-0.2], [-0.1], [0.1], [0.2], [0.4], [0.8]])
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    result = cv_performance(FixedScoreClassifier(), X, y, metric=metric, k=2)

    assert result == pytest.approx((1.0, 1.0, 1.0))


def test_cv_performance_handles_kernel_ridge_predictions_as_scores():
    X = np.array([[-4.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    result = cv_performance(
        KernelRidge(alpha=0.01, kernel="linear"), X, y, metric="accuracy", k=2
    )

    assert result == pytest.approx((1.0, 1.0, 1.0))


def test_unsupported_metric_is_rejected():
    X = np.array([[-1.0], [1.0]])
    y = np.array([-1, 1])

    with pytest.raises(ValueError, match="Unknown metric"):
        performance(FixedScoreClassifier(), X, y, metric="unsupported")

    with pytest.raises(ValueError, match="Unknown metric"):
        cv_performance(FixedScoreClassifier(), X, y, metric="unsupported", k=2)