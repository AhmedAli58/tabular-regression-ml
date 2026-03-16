"""
predict_target02.py
===================
Lightweight rule-based regression model for target02.

target02 depends on a small subset of the 273 input features and follows a
set of simple threshold rules, making it well-suited for a condition-action
predictor that runs on constrained hardware without any ML runtime.

Rules were discovered by fitting a depth-3 DecisionTreeRegressor on the
training data and translating the resulting splits into explicit conditions.

Key features used
-----------------
- feat_76  (index  76): primary split variable
- feat_173 (index 173): secondary split for the low feat_76 region
- feat_97  (index  97): secondary split for the high feat_76 region

Usage
-----
    python src/predict_target02.py --eval_file_path data/eval_features.csv
"""

import argparse
import operator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core condition-action framework
# ---------------------------------------------------------------------------

def cond_eval(condition, arr):
    """
    Evaluate a single threshold condition against a feature row.

    Parameters
    ----------
    condition : tuple or None
        A 3-tuple ``(feature_index, operator_str, threshold)``, e.g.
        ``(76, "<=", 0.50)``. Pass ``None`` for an unconditional branch.
    arr : numpy.ndarray
        1-D array of feature values for one sample.

    Returns
    -------
    bool
        ``True`` if the condition holds or if *condition* is ``None``.
    """
    ops = {
        ">":  operator.gt,
        ">=": operator.ge,
        "<":  operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True

    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


def framework(pairs, arr):
    """
    Apply a prioritised list of condition-action rules to every row of *arr*.

    Each rule is a (condition, calculation) pair. Rules are checked in order;
    the first matching condition fires its calculation and the rest are skipped.
    This mirrors a standard if / elif / else chain.

    Parameters
    ----------
    pairs : list of (condition, callable)
        - *condition*: a threshold tuple or ``None`` (matches everything).
        - *callable*: accepts a 1-D feature array, returns a scalar prediction.
    arr : numpy.ndarray
        2-D array of shape ``(n_samples, n_features)``.

    Returns
    -------
    list
        Predicted values, one per row in *arr*.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break

    return targets


# ---------------------------------------------------------------------------
# target02 prediction rules
# ---------------------------------------------------------------------------
# A depth-3 decision tree trained on the full training set identified three
# features that collectively explain ~36% of the variance in target02.
#
# Tree structure (leaf values are per-partition mean target02 on train set):
#
#   feat_76 <= 0.20
#       feat_173 <= 0.48  ->  0.18
#       feat_173 >  0.48  ->  0.55
#   feat_76 <= 0.50
#       feat_173 <= 0.46  ->  0.82
#       feat_173 >  0.46  ->  1.66
#   feat_76 >  0.50
#       feat_97  <= 0.29  ->   0.04
#       feat_97  <= 0.47  ->  -0.22
#       feat_97  <= 0.67  ->  -0.42
#       feat_97  >  0.67  ->  -0.72
# ---------------------------------------------------------------------------

def main(args):
    """
    Run the rule-based target02 predictor on the evaluation dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain ``eval_file_path``: path to the evaluation feature CSV.

    Returns
    -------
    list
        Predicted target02 values for every row in the evaluation file.
    """

    # Rule 1: low feat_76 region (feat_76 <= 0.20) — refine with feat_173
    condition_1 = (76, "<=", 0.20)

    def calc_1(arr):
        """Low feat_76 branch: threshold on feat_173 at 0.48."""
        return 0.18 if arr[173] <= 0.48 else 0.55

    # Rule 2: mid feat_76 region (0.20 < feat_76 <= 0.50) — refine with feat_173
    condition_2 = (76, "<=", 0.50)

    def calc_2(arr):
        """Mid feat_76 branch: threshold on feat_173 at 0.46."""
        return 0.82 if arr[173] <= 0.46 else 1.66

    # Rule 3: high feat_76 region (feat_76 > 0.50) — four-way split on feat_97
    def calc_3(arr):
        """High feat_76 branch: four-level threshold on feat_97."""
        if arr[97] <= 0.29:
            return 0.04
        elif arr[97] <= 0.47:
            return -0.22
        elif arr[97] <= 0.67:
            return -0.42
        else:
            return -0.72

    pair_list = [
        (condition_1, calc_1),
        (condition_2, calc_2),
        (None, calc_3),   # default: feat_76 > 0.50
    ]

    data_array = pd.read_csv(args.eval_file_path).values
    return framework(pair_list, data_array)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rule-based target02 predictor"
    )
    parser.add_argument(
        "--eval_file_path",
        required=True,
        help="Path to the evaluation feature CSV",
    )
    args = parser.parse_args()

    predictions = main(args)
    print(f"Generated {len(predictions)} predictions for target02.")
