import pytest
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression

from project1 import get_classifier


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_logistic_classifier_uses_penalty_and_other_arguments(penalty):
    class_weight = {-1: 1.5, 1: 2.0}

    classifier = get_classifier(
        loss="logistic",
        penalty=penalty,
        C=2.5,
        class_weight=class_weight,
    )

    assert isinstance(classifier, LogisticRegression)
    assert classifier.penalty == penalty
    assert classifier.C == 2.5
    assert classifier.class_weight == class_weight
    assert classifier.fit_intercept is False


def test_logistic_classifier_defaults_to_l2():
    classifier = get_classifier()

    assert isinstance(classifier, LogisticRegression)
    assert classifier.penalty == "l2"


def test_squared_error_classifier_uses_kernel_arguments():
    classifier = get_classifier(
        loss="squared_error",
        C=4.0,
        kernel="linear",
        gamma=0.25,
    )

    assert isinstance(classifier, KernelRidge)
    assert classifier.alpha == pytest.approx(1 / (2 * 4.0))
    assert classifier.kernel == "linear"
    assert classifier.gamma == 0.25