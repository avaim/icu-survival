# ICU Survival Prediction

A machine-learning project for predicting mortality risk from the first 48 hours of an ICU admission. Originally developed for **University of Michigan EECS 445 (Introduction to Machine Learning)**, then extended with **AI-assisted temporal feature engineering** to investigate whether changes in vital signs improve prediction.

> **Research/education only:** This is a course project and experimental extension, not a clinically validated model or a tool for medical decisions.

## Project overview

The original approach summarizes each patient's measurements from the first 48 hours (for example, maximum heart rate) and trains classification models to predict mortality. Summaries are useful, but a single maximum cannot show whether a vital sign increased or decreased during the admission.

The extension compares two feature sets using the **same patients and logistic regression settings**:

| Feature set | Features | Description |
| --- | ---: | --- |
| Baseline | 40 | Original patient-level and maximum-over-48-hours features |
| Enhanced | 65 | All baseline features plus 25 time-aware features |

For each of **HR, RespRate, Temp, MAP, and GCS**, the new extractor calculates:

- **Early mean:** average of valid measurements at 0 ≤ hour < 24
- **Late mean:** average of valid measurements at 24 ≤ hour ≤ 48
- **Change in mean:** late mean − early mean
- **Early missingness:** 1 if the early window has no valid readings; otherwise 0
- **Late missingness:** 1 if the late window has no valid readings; otherwise 0

The input uses elapsed-time `HH:MM` timestamps, including hour values greater than 23. Missing measurements marked `-1` are excluded from averages; absent means or differences are stored as `NaN` for later imputation. These indicators describe *entirely unavailable windows*, not the percentage of readings missing.

## Results

The experiment in `compare_models.py` uses the **first 2,000 labeled patients**, with `In-hospital_death` as the outcome (`-1` = survived, `1` = died). It evaluates a fixed, L2-regularized logistic regression (`C=1.0`, `liblinear`, no intercept). Both feature sets use **mean imputation and standard scaling fitted only on training data**.

### Five-fold stratified cross-validation

Both models use identical patient folds (400 validation patients per fold). Values are the **mean (population standard deviation)** across five folds.

| Metric | Baseline | Enhanced |
| --- | ---: | ---: |
| Accuracy | 0.6800 (0.0119) | 0.6835 (0.0215) |
| Precision | 0.2760 (0.0183) | 0.2860 (0.0167) |
| F1-score | 0.4068 (0.0285) | 0.4235 (0.0208) |
| AUROC | 0.7885 (0.0131) | **0.8103 (0.0076)** |
| Average precision | 0.4238 (0.0359) | 0.4530 (0.0322) |
| Sensitivity | 0.7742 (0.0684) | **0.8169 (0.0324)** |
| Specificity | 0.6643 (0.0144) | 0.6614 (0.0246) |

The enhanced feature set improved six of seven *average* metrics; specificity declined slightly. These are exploratory results on one dataset, **not proof of a statistically significant improvement or generalization to another hospital**.

### Original 80/20 split

A separate fixed 1,600-patient training / 400-patient test split gave **AUROC 0.7911 (baseline) vs. 0.7737 (enhanced)**, even though accuracy, F1 and sensitivity improved. This mixed result motivated the additional cross-validation experiment. The single-split test set has been inspected repeatedly, so it should not be treated as an untouched final evaluation.

## AI-assisted development and verification

The original course project was extended with an AI coding assistant for the following development tasks:

1. **Temporal feature implementation:** Helped build `temporal_features.py` to extract statistics from *raw timestamped measurements*, rather than trying to split an already-aggregated feature matrix. An older `split_time_windows()` attempt in `project1.py` is disabled; it is **not** used by the comparison experiment.
2. **Testing and inspection:** Helped write focused `pytest` cases for time-window boundaries, missing `-1` values, absent measurements and hand-computable examples, plus `inspect_patient.py` for displaying real patient readings.
3. **Model comparison:** Helped create `compare_models.py` with matched preprocessing and evaluation, then added five-fold cross-validation after the initial single-split results were mixed.

**My review and decisions:** I kept the original course code separate, reviewed the data flow and metric calculations, and manually checked extracted values against raw patient CSVs rather than relying exclusively on AI-generated tests. For example, for one patient's heart rate, the early and late means were **66.5417** and **78.6923** bpm, for a change of **+12.1506** bpm. I also compared both approaches using the same patients and folds, and retained results that did *not* favor the extension.

The new feature extraction and evaluation are reproducible; the model results should be interpreted as experimental evidence, not clinical conclusions.

## Repository guide

| Path | Purpose |
| --- | --- |
| `project1.py` | Original coursework: feature extraction, classifiers, evaluation and challenge workflow |
| `helper.py`, `test_output.py` | Course-supplied helper and submission-checking scripts |
| `config.yaml` | Original feature definitions and random seed |
| `temporal_features.py` | Additional early/late temporal features |
| `compare_models.py` | Matched baseline vs. enhanced experiments; single split and five-fold CV |
| `inspect_patient.py` | Raw-measurement output for manual verification |
| `tests/` | Automated unit tests |
| `data/labels.csv`, `data/files/` | Course dataset labels and per-patient measurement CSVs |
| `L0_Norm.png`, `feature_statistics.csv`, `avaim.csv` | Original coursework artifacts / generated outputs |

The main entry point for the extension is **`compare_models.py`**, not the original assignment's `project1.py` challenge workflow.

## Run locally

Use Python **3.12** and run commands from the repository root:

```bash
conda create -n icu-survival python=3.12
conda activate icu-survival
python -m pip install -r requirements.txt

# Run automated tests
python -m pytest -v

# Train and evaluate both feature sets
python compare_models.py

# Display source observations for manual feature checks
python inspect_patient.py
```

The comparison script expects `data/labels.csv` and corresponding `data/files/<RecordID>.csv` files. It can take time to load thousands of small patient files. If preparing a separate copy of this repository without the data, obtain the dataset through an appropriately authorized source before running the comparison.

## Limitations and possible next steps

- One institution/course dataset; performance outside this population has not been established.
- The extra temporal features summarize two broad windows, not the full trajectory or spacing between readings.
- A missingness flag indicates an entirely empty window, not the fraction of readings missing.
- Further evaluation should use an untouched external test cohort and assess calibration and clinical usefulness before any real-world application.
- The original course prediction workflow and the later 2,000-patient in-hospital-mortality comparison answer different experimental questions.

## Data and attribution

This project builds on the **EECS 445 ICU survival assignment** and its supplied starter helpers. The ICU measurements were provided for coursework and described in the original project as a subset of publicly available data associated with Beth Israel Deaconess Medical Center. Reuse and redistribution of the included dataset and starter materials should follow their applicable source and course terms.
