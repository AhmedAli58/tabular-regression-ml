# Tabular Machine Learning Regression Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

An end-to-end machine learning pipeline for predicting continuous target variables from high-dimensional tabular data. The project combines a Gradient Boosting ensemble model for maximum predictive accuracy with a lightweight rule-based model designed for interpretability and edge deployment — no ML runtime required at inference time.

The pipeline covers the full ML workflow: data loading, model training, evaluation, prediction generation, and reporting.

---

## Problem Description

The task involves predicting two continuous numerical targets (`target01`, `target02`) from a structured dataset of 273 numerical features across 10,000 samples.

- **target01** is predicted using a supervised Gradient Boosting Regressor, selected after benchmarking four candidate models.
- **target02** exhibits a low-dimensional structure — it depends on only three features and follows a set of simple threshold rules, making it suitable for a rule-based predictor that can run directly on constrained hardware.

---

## Project Structure

```
.
├── data/
│   ├── train_features.csv      # Training feature matrix (10,000 × 273)
│   ├── train_targets.csv       # Training targets: target01, target02
│   └── eval_features.csv       # Evaluation feature matrix (10,000 × 273)
│
├── predictions/
│   └── target01_predictions.csv  # Model predictions for target01
│
├── src/
│   ├── train_model.py          # Gradient Boosting training and prediction pipeline
│   ├── predict_target02.py     # Rule-based predictor for target02
│   └── generate_report.py      # PDF report generation
│
├── report/
│   └── project_report.pdf      # Full project report with methodology and results
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Methodology

### Gradient Boosting — target01

Four regression models were trained and evaluated on the held-out evaluation set:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Decision Tree (depth 3) | 0.1192 | 0.1053 | 0.190 |
| Ridge Regression | 0.1197 | 0.1055 | 0.184 |
| Random Forest | 0.1143 | 0.0975 | 0.255 |
| **Gradient Boosting** | **0.0740** | **0.0652** | **0.688** |

Gradient Boosting was selected for its strong generalisation, achieving a 38% reduction in RMSE over the next-best model. It builds an additive ensemble in a forward stage-wise manner — each tree corrects the residuals of the previous ensemble — which makes it highly effective at capturing non-linear feature interactions in tabular data.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `random_state` | 42 |

### Rule-Based Model — target02

target02 depends on just three features: `feat_76`, `feat_173`, and `feat_97`. A depth-3 decision tree was fitted to identify the optimal split thresholds, then translated into explicit threshold conditions:

```
if feat_76 <= 0.20:
    target02 = 0.18  if feat_173 <= 0.48  else  0.55
elif feat_76 <= 0.50:
    target02 = 0.82  if feat_173 <= 0.46  else  1.66
else:
    if   feat_97 <= 0.29:  target02 =  0.04
    elif feat_97 <= 0.47:  target02 = -0.22
    elif feat_97 <= 0.67:  target02 = -0.42
    else:                   target02 = -0.72
```

This approach requires no ML library at inference time, making it suitable for deployment on edge devices or embedded systems.

**Results:** RMSE = 0.4653 · MAE = 0.3419 · R² = 0.360

---

## Installation

Requires Python 3.10 or higher.

```bash
git clone https://github.com/AhmedAli58/tabular-regression-ml.git
cd tabular-regression-ml
pip install -r requirements.txt
```

---

## Running the Project

**Train the Gradient Boosting model and generate target01 predictions:**

```bash
python src/train_model.py
```

With explicit paths:

```bash
python src/train_model.py \
    --train_features data/train_features.csv \
    --train_targets  data/train_targets.csv  \
    --eval_features  data/eval_features.csv  \
    --output         predictions/target01_predictions.csv
```

**Run the rule-based target02 predictor:**

```bash
python src/predict_target02.py --eval_file_path data/eval_features.csv
```

**Regenerate the PDF report:**

```bash
python src/generate_report.py
```

---

## Outputs

Model predictions are saved to:

```
predictions/target01_predictions.csv
```

The file contains 10,000 rows with a single column `target01`, corresponding row-for-row to `data/eval_features.csv`.

A full project report covering methodology, model selection, results, and analysis is available at:

```
report/project_report.pdf
```

---

## Future Improvements

- **Feature engineering** — construct interaction terms and polynomial features to expose non-linear relationships
- **Hyperparameter tuning** — apply `RandomizedSearchCV` or Bayesian optimisation for a more thorough search
- **Cross-validation** — replace single held-out evaluation with k-fold CV for more robust performance estimates
- **Model stacking** — combine predictions from multiple base models using a meta-learner
- **Dimensionality reduction** — use feature importance scores or PCA to reduce the 273-feature space
- **Extended rule-based model** — apply the same rule-discovery approach to target01 for lightweight deployment



