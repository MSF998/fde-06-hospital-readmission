import pickle

import numpy as np
import pandas as pd

from src.features import CATEGORICAL_COLS, NUMERIC_COLS, build_preprocessor
from src.train import (
    always_no_accuracy,
    build_pipeline,
    calibration_bins,
    evaluate,
    recall_at_precision,
)


def make_xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.integers(0, 10, size=n).astype(float) for c in NUMERIC_COLS})
    for c in CATEGORICAL_COLS:
        X[c] = rng.choice(["a", "b", "c"], size=n)
    signal = X[NUMERIC_COLS[0]] + rng.normal(scale=2.0, size=n)
    y = (signal > np.quantile(signal, 0.85)).astype(int)
    return X, y


def test_feature_lists_do_not_include_ids_or_the_target():
    forbidden = {"readmit_30", "readmitted", "patient_nbr", "encounter_id"}
    assert forbidden.isdisjoint(NUMERIC_COLS)
    assert forbidden.isdisjoint(CATEGORICAL_COLS)
    assert set(NUMERIC_COLS).isdisjoint(CATEGORICAL_COLS)


def test_preprocessor_handles_missing_values_and_outputs_no_nan():
    X, _ = make_xy()
    X.loc[0, NUMERIC_COLS[0]] = np.nan
    X.loc[1, CATEGORICAL_COLS[0]] = np.nan
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    assert not np.isnan(out).any()


def test_pipeline_has_expected_structure():
    pipe = build_pipeline()
    assert list(pipe.named_steps) == ["prep", "clf"]
    assert pipe.named_steps["clf"].class_weight == "balanced"
    assert [name for name, _, _ in pipe.named_steps["prep"].transformers] == ["num", "cat"]


def test_fitted_pipeline_scores_between_0_and_1():
    X, y = make_xy()
    pipe = build_pipeline().fit(X, y)
    scores = pipe.predict_proba(X)[:, 1]
    assert scores.shape == (len(X),)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_fitted_pipeline_scores_an_unseen_category_without_error():
    X, y = make_xy()
    pipe = build_pipeline().fit(X, y)
    row = X.iloc[[0]].copy()
    row[CATEGORICAL_COLS[0]] = "never-seen-in-training"
    score = pipe.predict_proba(row)[0, 1]
    assert 0 <= score <= 1


def test_pickled_pipeline_does_not_depend_on_project_code():
    X, y = make_xy()
    blob = pickle.dumps(build_pipeline().fit(X, y))
    assert b"src." not in blob


def test_recall_at_precision_on_a_toy_example():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert recall_at_precision(y, s, min_precision=0.6) == 1.0
    assert recall_at_precision(y, s, min_precision=0.9) == 0.5


def test_recall_at_precision_is_zero_when_target_is_unreachable():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.7])
    assert recall_at_precision(y, s, min_precision=0.9) == 0.0


def test_always_no_accuracy_shows_why_accuracy_misleads():
    y = np.array([1] * 11 + [0] * 89)
    assert abs(always_no_accuracy(y) - 0.89) < 1e-9


def test_evaluate_returns_the_headline_metrics():
    y = np.array([0, 0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = evaluate(y, perfect, min_precision=0.5)
    assert set(result) == {"pr_auc", "recall_at_precision", "brier_score", "min_precision"}
    assert result["pr_auc"] == 1.0
    assert result["recall_at_precision"] == 1.0


def test_calibration_bins_shape_and_columns():
    rng = np.random.default_rng(0)
    s = rng.random(500)
    y = (rng.random(500) < s).astype(int)
    table = calibration_bins(y, s, n_bins=5)
    assert list(table.columns) == ["mean_predicted", "observed_rate"]
    assert 1 <= len(table) <= 5
