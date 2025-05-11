"""
EECS 445 Winter 2025

This script should contain most of the work for the project. You will need to fill in every TODO comment.
"""


import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import helper


__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)


def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    # Replace unknown values with np.nan
    df_replaced = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    # extract raw values of time-invariant variables into feature dict
    for _, row in static.iterrows():
        feature_dict[row["Variable"]] = row["Value"]

    # extract max of time-varying variables into feature dict
    for variable in timeseries["Variable"].unique():
        max_value = timeseries[timeseries["Variable"] == variable]["Value"].max()
        feature_dict[f"max_{variable}"] = max_value
    
    return feature_dict


def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: array of shape (N, d) which could contain missing values
        
    Returns:
        X: array of shape (N, d) without missing values
    """

    # iterate over each column
    for col in range(X.shape[1]):
        # compute the mean of the column and ignore np.nan values
        col_mean = np.nanmean(X[:, col])

        # replace np.nan values in the column with the mean
        X[:, col][np.isnan(X[:, col])] = col_mean

    return X


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: array of shape (N, d).

    Returns:
        X: array of shape (N, d). Values are normalized per column.
    """
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    ranges = max_vals - min_vals

    # for cols with max == min
    ranges[ranges == 0] = 1.0

    X_norm = (X - min_vals) / ranges
    return X_norm


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but highly recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(
            penalty="l2",
            C=C,
            solver="liblinear",
            fit_intercept=False,
            class_weight=class_weight,
            random_state=42,
        )
    elif loss == "squared_error":
        return KernelRidge(alpha=1 / (2 * C), kernel=kernel, gamma=gamma)
    
    raise ValueError(f"Unsupported loss function: {loss}")


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy"
) -> dict[str, float]:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X.
    Returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.

    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
    Returns:
        peformance for the specific metric
    """
    
    y_pred = clf_trained.predict(X)
    y_scores = clf_trained.decision_function(X)

    # Initialize metrics dictionary
    metrics = {}

    # Compute all metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    try:
        metrics["precision"] = precision_score(y_true, y_pred, pos_label=1)
    except ZeroDivisionError:
        metrics["precision"] = 0.0  # Handle divide-by-zero

    try:
        metrics["f1_score"] = f1_score(y_true, y_pred, pos_label=1)
    except ZeroDivisionError:
        metrics["f1_score"] = 0.0  # Handle divide-by-zero

    metrics["auroc"] = roc_auc_score(y_true, y_scores)
    metrics["average_precision"] = average_precision_score(y_true, y_scores)

    # Sensitivity and Specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["sensitivity"] = tp / (tp + fn)  # Sensitivity (Recall)
    metrics["specificity"] = tn / (tn + fp)  # Specificity

    return metrics


def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    
    # initialize cross-validation
    skf = StratifiedKFold(n_splits=k, shuffle=False)

    # store performance scores for each fold
    scores = []

    for train_index, test_index in skf.split(X, y):
        # split data into training and test sets for each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # train classifier
        clf.fit(X_train, y_train)

        # predict on test set
        if isinstance(clf, KernelRidge):
            y_scores = clf.predict(X_test)
        
        else:
            y_scores = clf.decision_function(X_test)

        #compute performance metric
        if metric == "accuracy":
            y_pred = np.where(y_scores >= 0, 1, -1)
            score = accuracy_score(y_test, y_pred)
        elif metric == "precision":
            y_pred = np.where(y_scores >= 0, 1, -1)
            try:
                score = precision_score(y_test, y_pred, pos_label=1)
            except ZeroDivisionError:
                score = 0.0
        elif metric == "f1_score":
            y_pred = np.where(y_scores >= 0, 1, -1)
            try:
                score = f1_score(y_test, y_pred, pos_label=1)
            except ZeroDivisionError:
                score = 0.0
        elif metric == "auroc":
            score = roc_auc_score(y_test, y_scores)
        elif metric == "average_precision":
            score = average_precision_score(y_test, y_scores)
        elif metric == "sensitivity":
            y_pred = np.where(y_scores >= 0, 1, -1)
            cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
            tn, fp, fn, tp = cm.ravel()
            score = tp / (tp + fn)
        elif metric == "specificity":
            y_pred = np.where(y_scores >= 0, 1, -1)
            cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
            tn, fp, fn, tp = cm.ravel()
            score = tn / (tn + fp)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)

    mean_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
        
    return mean_score, min_score, max_score


def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    
    if not C_range:
        C_range = [10**i for i in range (-3, 4)]
    
    best_C = None
    best_penalty = None
    best_score = -1

    for C in C_range:
        for penalty in penalties:

            # initialize
            clf = LogisticRegression(
                penalty=penalty,
                C=C,
                solver="liblinear",

                fit_intercept=False,
                random_state=42
            )

            # compute cross-validation performance
            mean_score, _, _ = cv_performance(clf, X, y, metric=metric, k=k)

            # update best hyperparameters if combination is better
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_penalty = penalty

    return best_C, best_penalty


def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    # default ranges for C and gamma
    if not C_range:
        C_range = [10**i for i in range(-3, 4)]
    if not gamma_range:
        gamma_range = [10**i for i in range(-3, 4)]

    # initialization
    best_C = None
    best_gamma = None
    best_score = -1

    # grid search over C and gamma
    for C in C_range:
        for gamma in gamma_range:
            # initialization
            clf = KernelRidge(
                alpha=1 / (2 * C),
                kernel="rbf",
                gamma=gamma,
            )

            # cross-validation performance
            mean_score, _, _ = cv_performance(clf, X, y, metric=metric, k=k)

            # update best hyperparameters if necessary

            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_gamma = gamma

    return best_C, best_gamma


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}

    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    plt.figure(figsize=(10, 6))

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # initialize clf with C and penalty
            clf = LogisticRegression(
                penalty=penalty,
                C=C,
                solver="liblinear",
                fit_intercept=False,
                random_state=42,
            )
            
            # fit clf to X and y
            clf.fit(X, y)
            
            # extract learned coefficients from clf into w
            w = clf.coef_.flatten() # to get a 1D array
            
            # count the number of nonzero coefficients and append the count to norm0
            non_zero_count = np.sum(w != 0)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()


def main():
    print(f"Using Seed = {seed}")
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED IMPLEMENTING generate_feature_vector,
    #       fill_missing_values AND normalize_feature_matrix!
    # NOTE: Only set debug=True when testing your implementation against debug.txt. DO NOT USE debug=True when
    #       answering the project questions! It only loads a small sample (n = 100) of the data in debug mode,
    #       so your performance will be very bad when working with the debug data.
    # X_train, y_train, X_test, y_test, feature_names = helper.get_project_data(debug=False)

    X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data()

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # splitting up the 48-hr window
    def split_time_windows(X):
        n_samples, n_features = X.shape
        n_time_steps = 48
        n_features_per_step = n_features // n_time_steps

        X_window1 = X[:, :n_features_per_step * 24] # first 24 hrs
        X_window2 = X[:, n_features_per_step * 24] # last 24 hrs

        # both windows need same number of dimensions
        if X_window1.ndim == 1:
            X_window1 = X_window1.reshape(-1, 1)
        if X_window2.ndim == 1:
            X_window2 = X_window2.reshape(-1, 1)

        return np.hstack([X_window1, X_window2])
    
    X_challenge = split_time_windows(X_challenge)
    X_heldout = split_time_windows(X_heldout)

    # use summary statistics for numerical variables
    def compute_summary_statistics(X):
        mean = np.mean(X, axis=1, keepdims=True)
        median = np.median(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return np.hstack([mean, median, std])
    
    X_challenge = np.hstack([X_challenge, compute_summary_statistics(X_challenge)])
    X_heldout = np.hstack([X_heldout, compute_summary_statistics(X_heldout)])

    # standardize features
    scaler = StandardScaler()
    X_challenge = scaler.fit_transform(X_challenge)
    X_heldout = scaler.transform(X_heldout)

    # use GridSearchCV to find the best C
    param_grid = {
        "logreg__C": [0.01, 0.1, 1.0, 10, 100],
        "logreg__penalty": ["l1", "l2"],
    }

    # use logistic regression
    pipline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear", fit_intercept=False, random_state=seed)),
    ])

    # use 5-fold cross-validation with grid search
    grid_search = GridSearchCV(pipline, param_grid, cv=5, scoring="roc_auc")
    grid_search.fit(X_challenge, y_challenge)

    # use the best model
    best_clf = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    y_label = best_clf.predict(X_heldout).astype(int) # binary predictions
    y_score = best_clf.decision_function(X_heldout) # risk scores

    helper.save_challenge_predictions(y_label, y_score, "avaim")

    print("Predictions saved to avaim.csv")

    y_pred_challenge = best_clf.predict(X_challenge)
    cm = confusion_matrix(y_challenge, y_pred_challenge, labels=[-1, 1])
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
