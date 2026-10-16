import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline

from src.features import CATEGORICAL_COLS, NUMERIC_COLS, build_preprocessor

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
MODEL_PATH = "models/model.joblib"


def build_pipeline():
    """Pipeline with steps named "prep" and "clf"."""
    # One object = preprocessing + model. fit() learns everything from the data
    # it is given (train only); predict_proba() just applies what was learned.
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            # "balanced": rare positives count more, so the model tries to find them.
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def recall_at_precision(y_true, scores, min_precision: float) -> float:
    """Highest recall among thresholds whose precision >= min_precision; 0.0 if none."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    # The last point (precision 1, recall 0) is a placeholder, so it adds nothing
    # to the max. If no threshold reaches the target, the mask is empty -> 0.0.
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def always_no_accuracy(y_true) -> float:
    """Accuracy of a model that predicts 0 for everyone."""
    return float((np.asarray(y_true) == 0).mean())


def calibration_bins(y_true, scores, n_bins: int = 5) -> pd.DataFrame:
    """Columns: mean_predicted, observed_rate."""
    observed, predicted = calibration_curve(y_true, scores, n_bins=n_bins)
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed})


def evaluate(y_true, scores, min_precision: float = 0.2) -> dict:
    """Keys: pr_auc, recall_at_precision, brier_score, min_precision."""
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_precision": recall_at_precision(y_true, scores, min_precision),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "min_precision": min_precision,
    }


def main() -> None:
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)  # test.parquet is never touched here

    features = NUMERIC_COLS + CATEGORICAL_COLS
    X_train, y_train = train[features], train["readmit_30"]
    X_val, y_val = val[features], val["readmit_30"]

    pipe = build_pipeline().fit(X_train, y_train)  # learns from train only
    scores = pipe.predict_proba(X_val)[:, 1]

    print("base rate (random model PR-AUC):", round(y_val.mean(), 4))
    print("validation metrics:", evaluate(y_val, scores))
    print("always-no accuracy:", round(always_no_accuracy(y_val), 4))
    print(calibration_bins(y_val, scores))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)


if __name__ == "__main__":
    main()
