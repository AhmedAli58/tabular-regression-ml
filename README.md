# Tabular Machine Learning Regression Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange)
![pandas](https://img.shields.io/badge/pandas-2.2.1-150458)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project solves two regression tasks on anonymized tabular data with 273 numerical features.

For `target01`, I compared four regression models and selected Gradient Boosting as the final model.  
For `target02`, I extracted simple threshold-based rules from a shallow decision tree to create a lightweight predictor without an ML runtime.

## Approach

### target01
Four models were benchmarked:

- Decision Tree
- Ridge Regression
- Random Forest
- Gradient Boosting

Gradient Boosting performed best and was selected for final prediction generation.

### target02
A depth-3 decision tree was used to identify the smallest set of features and thresholds needed to predict `target02`.

The final rule-based predictor uses only:

- `feat_76`
- `feat_173`
- `feat_97`

## Results

### target01

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Decision Tree (depth 3) | 0.1192 | 0.1053 | 0.190 |
| Ridge Regression | 0.1197 | 0.1055 | 0.184 |
| Random Forest | 0.1143 | 0.0975 | 0.255 |
| **Gradient Boosting** | **0.0740** | **0.0652** | **0.688** |

### target02

    if feat_76 <= 0.20:
        target02 = 0.18 if feat_173 <= 0.48 else 0.55
    elif feat_76 <= 0.50:
        target02 = 0.82 if feat_173 <= 0.46 else 1.66
    else:
        if feat_97 <= 0.29:
            target02 = 0.04
        elif feat_97 <= 0.47:
            target02 = -0.22
        elif feat_97 <= 0.67:
            target02 = -0.42
        else:
            target02 = -0.72

Rule-based model performance:

- `RMSE = 0.4653`
- `MAE = 0.3419`
- `R² = 0.360`
