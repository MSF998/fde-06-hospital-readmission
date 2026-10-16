import os
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

REQUESTS = Counter("predict_requests_total", "Total /predict requests", ["status"])
LATENCY = Histogram("predict_latency_seconds", "Latency of /predict in seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.version = os.environ.get("MODEL_VERSION", "baseline-lr-v1")
    path = os.environ.get("MODEL_PATH", "models/model.joblib")
    try:
        app.state.model = joblib.load(path)
    except FileNotFoundError:
        print(f"WARNING: model file not found at {path}; serving 503 until it exists")
    yield


app = FastAPI(title="Readmission risk", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: dict[str, Any]


def _categorical_columns(model) -> list[str]:
    for name, _, cols in model.named_steps["prep"].transformers_:
        if name == "cat":
            return list(cols)
    return []


def _build_row(model, features: dict[str, Any]) -> pd.DataFrame:
    columns = list(model.feature_names_in_)
    row = pd.DataFrame([{c: features[c] for c in columns}])
    # JSON null -> real NaN, so the imputers see it as missing
    row = row.where(row.notna(), np.nan)
    # category codes may arrive as ints; the model was trained on strings
    for c in _categorical_columns(model):
        row[c] = row[c].map(lambda v: v if pd.isna(v) else str(v))
    return row


def _top_factors(model, row: pd.DataFrame, k: int = 5) -> list[dict]:
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    x = prep.transform(row)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contribution = x[0] * clf.coef_[0]
    names = prep.get_feature_names_out()
    order = np.argsort(-np.abs(contribution))[:k]
    return [{"feature": str(names[i]), "contribution": float(contribution[i])} for i in order]


@app.get("/health")
def health():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    status = "200"
    try:
        model = app.state.model
        if model is None:
            status = "503"
            raise HTTPException(status_code=503, detail="model not loaded")
        missing = [c for c in model.feature_names_in_ if c not in req.features]
        if missing:
            status = "422"
            raise HTTPException(status_code=422, detail=f"missing features: {missing}")
        row = _build_row(model, req.features)
        try:
            score = float(model.predict_proba(row)[0, 1])
            factors = _top_factors(model, row)
        except (ValueError, TypeError) as exc:
            status = "422"
            raise HTTPException(status_code=422, detail=f"could not score input: {exc}")
        return {"risk_score": score, "model_version": app.state.version, "top_factors": factors}
    finally:
        REQUESTS.labels(status=status).inc()
        LATENCY.observe(time.perf_counter() - start)
