# ICU Survival Prediction

**EECS 445 · Introduction to Machine Learning · University of Michigan**

A machine learning project that uses patient measurements from the first 48 hours of an ICU admission to predict mortality risk. The original course project covers feature engineering, model selection, class imbalance, and evaluation; I later revisited it and used an AI coding assistant to explore an improvement to the existing feature representation.

> **Educational project only.** These models are not clinically validated and are not intended for medical decision-making.

## Project overview

Electronic health records contain measurements that may help identify patients at risk of poor outcomes. In this project, I built and evaluated classification models on ICU admissions using demographic and clinical observations such as age, heart rate, blood pressure, temperature, and laboratory values.

The work has two related prediction tasks:

- **In-hospital mortality:** Predict whether a patient survives their hospital stay. This is the primary target for the project's model comparisons and feature-engineering experiments.
- **30-day mortality:** Train a classifier for the course's final challenge and produce predictions and risk scores for a held-out set.

Beyond obtaining a prediction, the project examines how preprocessing, regularization, model choice, and class imbalance affect performance.

## Dataset

The course dataset contains **12,000 ICU admissions** drawn from a publicly available research dataset associated with Beth Israel Deaconess Medical Center and PhysioNet. Each patient admission has:

- A CSV file in `data/files/`, with `Time`, `Variable`, and `Value` columns for observations made during the first **48 hours**
- A `RecordID` connecting the observations to `data/labels.csv`
- Two mortality outcomes, when available: `In-hospital_death` and `30-day_mortality` (`1` = died, `-1` = survived)

The original modeling exercises use **2,000 patients** (1,600 training and 400 testing). The final challenge uses **10,000 labeled admissions** and predicts outcomes for the remaining **2,000 held-out admissions**.

## Original project: methods and experiments

### 1. Feature extraction and preprocessing

I converted each patient's irregular, timestamped records into a fixed-length **40-feature vector**:

- Keep the recorded values of time-invariant variables
- Summarize each time-varying variable by its **maximum** value during the first 48 hours
- Treat unknown observations encoded as `-1` as missing (`NaN`)
- Impute missing values using feature-wise means
- Normalize features to help make model training and regularization more consistent across different measurement scales

### 2. Logistic regression and hyperparameter selection

I implemented model evaluation and **five-fold stratified cross-validation** to explore regularized logistic regression, including:

- **L1 vs. L2 regularization**
- The effect of regularization strength `C`
- Selection of hyperparameters using different performance metrics
- How coefficient sparsity changes with regularization (`L0_Norm.png`)

### 3. Class imbalance and evaluation

Because fewer patients die than survive, accuracy alone can be misleading. I investigated **class-weighted logistic regression** and the tradeoff between identifying patients who died and correctly identifying survivors.

Models were evaluated with **accuracy, precision, F1-score, AUROC, average precision, sensitivity, and specificity**, including analysis of ROC curves and decision thresholds.

### 4. Ridge regression and nonlinear models

I compared logistic regression with **kernel ridge regression**, examining both linear and **radial basis function (RBF)** kernels. I used cross-validation to explore the effect of `C` and the RBF kernel parameter `gamma` on model performance.

### 5. Final mortality prediction challenge

For the original course challenge, I built a logistic-regression prediction pipeline for **30-day mortality** using the larger training set, feature engineering, scaling, and hyperparameter search. The workflow exports binary predictions and continuous risk scores to `avaim.csv` for the held-out admissions.

## Later extension: AI-assisted feature engineering

### The challenge

The original 40-feature representation uses the **maximum** of each time-varying measurement. That describes how high a measurement became but discards information about **when** it occurred or whether it changed during the patient's ICU stay.

I revisited the project to test whether adding simple time-aware features would improve mortality prediction.

### What I implemented

With help from an AI coding assistant, I added `temporal_features.py`, which processes raw patient observations *before* their timestamps are discarded. For **heart rate, respiratory rate, temperature, mean arterial pressure, and Glasgow Coma Scale**, it creates five features:

1. Mean during **0–24 hours**
2. Mean during **24–48 hours**
3. Change in mean (**late − early**)
4. Indicator that the early window has no valid measurements
5. Indicator that the late window has no valid measurements

This adds **25 temporal features**, bringing the enhanced feature set from **40 to 65 features**.

I used AI to help write the extractor, generate unit tests and inspection tools, and build a baseline-versus-enhanced evaluation script. I reviewed the resulting code and **manually checked feature calculations against raw patient CSVs** rather than relying only on generated tests. For example, one patient's average heart rate increased from **66.54 to 78.69 bpm** across the two windows.

### Did the improvement help?

In `compare_models.py`, I compared logistic regression using the **original 40 features** against the **65-feature extension**, holding patient splits and model settings constant. The comparison fits mean imputation and standard scaling on training data only.

| Five-fold CV metric | Original | With temporal features |
| --- | ---: | ---: |
| AUROC | 0.7885 | **0.8103** |
| F1-score | 0.4068 | **0.4235** |
| Sensitivity | 0.7742 | **0.8169** |
| Specificity | **0.6643** | 0.6614 |

These are mean scores across five stratified folds on the 2,000-patient in-hospital mortality dataset. Six of seven tracked metrics improved on average, while specificity declined slightly. An earlier single 80/20 split gave a *lower* AUROC for the enhanced model (0.7737 vs. 0.7911), which motivated the broader cross-validation comparison.

The results suggest that temporal features may be useful on this dataset, but they do **not** establish statistical significance, clinical validity, or performance at another hospital.

## Repository structure

| File / folder | Purpose |
| --- | --- |
| `project1.py` | Original course implementation: feature processing, classifiers, metrics, parameter search, and challenge workflow |
| `helper.py`, `test_output.py` | Course-provided data-loading and submission-checking utilities |
| `config.yaml` | Feature definitions and random seed |
| `temporal_features.py` | AI-assisted temporal feature extraction |
| `compare_models.py` | Controlled comparison of original and enhanced feature sets |
| `inspect_patient.py` | Displays raw observations and temporal features for manual checking |
| `tests/` | Automated tests for original functionality and temporal extraction |
| `data/` | Patient measurements and labels |
| `L0_Norm.png`, `feature_statistics.csv`, `avaim.csv` | Coursework analysis and prediction outputs |

The older `split_time_windows()` attempt in `project1.py` is disabled because it operates on features that have already been aggregated. The later extension uses timestamps directly in `temporal_features.py`; the two implementations should not be confused.

## Running the project

The recent extension was developed using Python **3.12**. From the repository root:

```bash
conda create -n icu-survival python=3.12
conda activate icu-survival
python -m pip install -r requirements.txt

# Run automated tests
python -m pytest -v

# Compare original and enhanced feature sets
python compare_models.py

# Inspect raw patient measurements and generated features
python inspect_patient.py

# Run the original challenge workflow
python project1.py
```

The scripts expect `data/labels.csv` and patient files under `data/files/`. The challenge workflow writes `avaim.csv`; running it again may overwrite that output.

## Limitations and future work

This is an educational ML experiment, not a clinical risk model. Patient observations are irregular, the original features are coarse summaries, and the added temporal features capture only two broad time windows. Further work could investigate finer-grained trends, alternative missing-data strategies, calibration, and validation on an independent patient cohort.

**Attribution:** The project was originally developed for University of Michigan EECS 445 using course-provided scaffolding and ICU data described by the assignment as sourced from PhysioNet. The later temporal feature extension and its evaluation were added separately. Redistribution of any course materials and patient-level data should follow the applicable permissions and licenses.
