# Tabular Machine Learning Regression Pipeline

An end-to-end regression pipeline for predicting continuous variables from high-dimensional tabular data. The project covers model comparison, ensemble learning, and rule-based prediction — combining a Gradient Boosting model with an interpretable threshold-based system.

---

## Features

- End-to-end regression pipeline on tabular data (273 features, 10k samples)
- Systematic model comparison across four algorithms
- Gradient Boosting Regressor for maximum predictive accuracy
- Rule-based prediction system deployable without any ML runtime
- Clean, reproducible, well-documented code

---

## Project Structure

```
.
├── data/
│   ├── train_features.csv       # Training feature matrix (10,000 x 273)
│   ├── train_targets.csv        # Training targets: target01, target02
│   └── eval_features.csv        # Evaluation feature matrix (10,000 x 273)
│
├── predictions/
│   └── target01_predictions.csv # Model output for target01
│
├── src/
│   ├── train_model.py        # Train GB model and generate predictions
│   ├── predict_target02.py      # Rule-based predictor for target02
│   └── generate_report.py       # Generate report/project_report.pdf
│
├── report/
│   └── project_report.pdf       # Full project report
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

**Train target01 model and generate predictions:**
```bash
python src/train_model.py
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

## Results

| Task | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| target01 | Gradient Boosting | 0.0740 | 0.0652 | 0.688 |
| target02 | Rule-based model | 0.4653 | 0.3419 | 0.360 |

---

## Model Details

### target01 — Gradient Boosting Regressor

Four models were benchmarked on the held-out evaluation set:

| Model | RMSE | R² |
|---|---|---|
| Decision Tree (depth 3) | 0.1192 | 0.190 |
| Ridge Regression | 0.1197 | 0.184 |
| Random Forest | 0.1143 | 0.255 |
| **Gradient Boosting** | **0.0740** | **0.688** |

Final hyperparameters: `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`

### target02 — Rule-Based Model

target02 depends on just three features (`feat_76`, `feat_173`, `feat_97`). Rules were discovered using a depth-3 decision tree and encoded as explicit threshold conditions — no ML library needed at inference time.

```
if feat_76 <= 0.20:
    target02 = 0.18  if feat_173 <= 0.48  else  0.55
elif feat_76 <= 0.50:
    target02 = 0.82  if feat_173 <= 0.46  else  1.66
else:
    if   feat_97 <= 0.29: target02 =  0.04
    elif feat_97 <= 0.47: target02 = -0.22
    elif feat_97 <= 0.67: target02 = -0.42
    else:                  target02 = -0.72
```

---

## Outputs

Predictions are written to:

```
predictions/target01_predictions.csv
```

---

## Future Improvements

- Cross-validated hyperparameter search (`RandomizedSearchCV`) for target01
- Feature importance analysis to reduce dimensionality
- Explore XGBoost / LightGBM as faster alternatives
- SHAP values for model interpretability
- Feature engineering and interaction terms

---

## References

1. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
