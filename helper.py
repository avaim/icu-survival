"""
EECS 445 Winter 2025

This script contains helper functions to load and preprocess the data for this project.
"""


import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import project1


__all__ = ["get_project_data", "get_challenge_data", "save_challenge_predictions"]


def load_features(split: str, n_jobs: int = -1) -> tuple[npt.NDArray, pd.DataFrame, list[str]]:
    """Use project1 functions to load and preprocess the feature vectors for a given split.

    Args:
        split: What subset of data indices to load.
        n_jobs: How many CPU cores to use when multiprocessing; defaults to all available cores.

    Returns:
        Tuple of the feature matrix, label dataframe, and feature names.
    """
    
    # get the indices of the correct split
    df_labels = pd.read_csv("data/labels.csv")
    match split:
        case "debug":
            # a random sample of 500 of the first 2000 datapoints in the training set
            n_debug = 500
            debug_indices = np.random.default_rng(445).choice(2_000, size=n_debug, replace=False)
            df_labels = df_labels.iloc[debug_indices]
        case "training_subset":
            df_labels = df_labels[:2_000]
        case "training_full":
            df_labels = df_labels[:10_000]
        case "challenge":
            df_labels = df_labels[10_000:]
        case _:
            raise ValueError(f"Invalid split \"{split}\"")
    
    def process_data(index: int) -> pd.DataFrame:
        """Helper function to process a single yaml data file in parallel."""
        return project1.generate_feature_vector(pd.read_csv(f"data/files/{index}.csv"))
    
    # load the feature vectors into a DataFrame in parallel
    features_df = pd.DataFrame(Parallel(n_jobs=n_jobs)(
        delayed(process_data)(i)
        for i in tqdm(df_labels["RecordID"], desc=f"Loading {split} data")
    ))
    # sort feature columns alphabetically by name
    features_df = features_df.sort_index(axis=1)
    print(f"Loaded n = {len(features_df)} feature vectors")
    
    # process the feature DataFrame using the project1 functions
    X = features_df.values
    X = project1.impute_missing_values(X)
    X = project1.normalize_feature_matrix(X)
    
    return X, df_labels, features_df.columns.tolist()


def get_project_data(
    debug: bool = False
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, list[str]]:
    """Load the training and testing dataset.
    
    This function does the following steps:
        1. Reads in the data labels from data/labels.csv, and determines which files to load.
        2. Use the project1 functions to generate a feature vector for each example.
        3. Aggregate the feature vectors into a feature matrix.
        4. Use the project1 functions to impute missing datapoints and normalize the data with respect to the
           population.
        5. Split the data into 80% training and 20% testing splits stratified based on the label.
    
    The labels for the dataset are y = {-1, 1}, where -1 indicates that the patient survived and 1 indicates
    that the patient died in the hospital.

    Args:
        debug: Whether to load the debug data instead of the normal data. The debug data should only be used
               to ensure that your algorithms are working as expected by comparing your results to debug.txt.
               Do NOT use the debug flag when answering any of the questions, use it ONLY for testing.

    Returns:
        Tuple of X_train, X_test, y_train, y_test, and feature_names.
    """
    
    X, df_labels, feature_names = load_features("debug" if debug else "training_subset")
    y = df_labels["In-hospital_death"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    return X_train, y_train, X_test, y_test, feature_names


def get_challenge_data() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, list[str]]:
    """Read the data for the challenge section of the project.
    
    This function is identical to get_project_data, except that it returns a different label for y_train (
    30-day_mortality instead of In-hospital_death) and does not return y_challenge as this is what you will be
    graded on.
    
    Returns:
        Tuple of X_train, X_challenge, y_train, and feature_names.
    """
    
    X_train, df_labels_train, feature_names = load_features("training_full")
    y_train = df_labels_train["30-day_mortality"].values
    X_challenge, _, _ = load_features("challenge")
    return X_train, y_train, X_challenge, feature_names


def save_challenge_predictions(y_label: npt.NDArray, y_score: npt.NDArray, uniqname: str) -> None:
    """
    Saves the challenge predictions to a CSV file named `uniqname.csv`.

    IMPORTANT: Ensure the order of test examples in the held-out challenge set remains unchanged, as this file
    will be used to evaluate your classifier.

    Args:
        y_label: Binary predictions from the linear classifier.
        y_score: Raw scores from the linear classifier.
        uniqname: Your uniqname to name the output file.
    """
    
    pd.DataFrame({"label": y_label, "risk_score": y_score}).to_csv(f"{uniqname}.csv", index=False)
