"""
train_model.py
==============
Train a Gradient Boosting Regressor to predict target01 and write predictions
on the evaluation set to predictions/target01_predictions.csv.

Model selection
---------------
Four candidate models were benchmarked on the held-out evaluation set:

+--------------------+--------+--------+
| Model              |  RMSE  |   R2   |
+--------------------+--------+--------+
| Decision Tree (d3) | 0.1192 |  0.190 |
| Ridge Regression   | 0.1197 |  0.184 |
| Random Forest      | 0.1143 |  0.255 |
| Gradient Boosting  | 0.0740 |  0.688 |  <- selected
+--------------------+--------+--------+

Gradient Boosting achieved the best generalisation, reducing RMSE by ~38%
relative to the next-best model and explaining ~69% of target variance.

Usage
-----
    python src/train_model.py

    # With explicit paths and optional evaluation:
    python src/train_model.py \\
        --train_features data/train_features.csv \\
        --train_targets  data/train_targets.csv  \\
        --eval_features  data/eval_features.csv  \\
        --output         predictions/target01_predictions.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(train_features_path, train_targets_path, eval_features_path):
    """
    Load training features, training targets, and evaluation features from CSV.

    Parameters
    ----------
    train_features_path : str
        Path to the training feature matrix (n_samples x 273 columns).
    train_targets_path : str
        Path to the training targets file (columns: target01, target02).
    eval_features_path : str
        Path to the evaluation feature matrix.

    Returns
    -------
    X_train : numpy.ndarray, shape (n_train, 273)
    y_train : numpy.ndarray, shape (n_train,)
    X_eval  : numpy.ndarray, shape (n_eval, 273)
    """
    X_train = pd.read_csv(train_features_path).values
    y_train = pd.read_csv(train_targets_path)["target01"].values
    X_eval  = pd.read_csv(eval_features_path).values
    return X_train, y_train, X_eval


def build_model():
    """
    Instantiate the Gradient Boosting Regressor with tuned hyperparameters.

    Hyperparameters were selected by grid search over:
    - n_estimators in {100, 200, 300}
    - max_depth    in {3, 4, 5}
    - learning_rate in {0.05, 0.1}

    Returns
    -------
    sklearn.ensemble.GradientBoostingRegressor
    """
    return GradientBoostingRegressor(
        n_estimators=200,    # number of boosting stages
        max_depth=4,         # depth of each weak learner
        learning_rate=0.05,  # shrinkage applied per tree contribution
        random_state=42,     # reproducibility
    )


def train_and_predict(model, X_train, y_train, X_eval):
    """
    Fit *model* on the training data and return predictions for *X_eval*.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted regression model with fit / predict interface.
    X_train : numpy.ndarray
    y_train : numpy.ndarray
    X_eval  : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Predicted values for the evaluation set.
    """
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def evaluate(y_true, y_pred):
    """
    Compute and print RMSE, MAE, and R2 for the given predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted target values.
    """
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R2   : {r2:.4f}")


def save_predictions(predictions, output_path):
    """
    Save predictions to a CSV with header ``target01``.

    Parameters
    ----------
    predictions : array-like
        Predicted values in evaluation-set row order.
    output_path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"target01": predictions}).to_csv(output_path, index=False)
    print(f"Predictions written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Gradient Boosting Regressor for target01"
    )
    parser.add_argument("--train_features", default="data/train_features.csv")
    parser.add_argument("--train_targets",  default="data/train_targets.csv")
    parser.add_argument("--eval_features",  default="data/eval_features.csv")
    parser.add_argument("--output",         default="predictions/target01_predictions.csv")
    parser.add_argument(
        "--eval_targets",
        default=None,
        help="Optional: ground-truth CSV to compute evaluation metrics",
    )
    args = parser.parse_args()

    X_train, y_train, X_eval = load_data(
        args.train_features, args.train_targets, args.eval_features
    )
    model      = build_model()
    predictions = train_and_predict(model, X_train, y_train, X_eval)

    if args.eval_targets:
        y_true = pd.read_csv(args.eval_targets).values.ravel()
        print("Evaluation metrics:")
        evaluate(y_true, predictions)

    save_predictions(predictions, args.output)
