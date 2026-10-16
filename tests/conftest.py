import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@pytest.fixture
def tiny_model_path(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"n1": rng.normal(size=300), "c1": rng.choice(["1", "2", "3"], size=300)}
    )
    y = ((X["n1"] + (X["c1"] == "2") + rng.normal(scale=0.5, size=300)) > 1).astype(int)
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                ["n1"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["c1"],
            ),
        ]
    )
    pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(class_weight="balanced"))]).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def client(tiny_model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_model_path))
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from api.main import app

    with TestClient(app) as c:
        yield c
