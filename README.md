# 30-day hospital readmission risk (Pass 1)

Predict, at discharge, whether a diabetic patient will be readmitted within 30 days. Pass 1 is a thin end-to-end slice: data -> baseline model -> API -> Docker + Prometheus. See `docs/superpowers/specs/` for the design and `docs/decisions.md` for the choices made and why.

## Baseline results (validation set)

Logistic regression with `class_weight="balanced"`, trained on train only. The test set has not been used.

| Metric | Value | How to read it |
|---|---|---|
| PR-AUC | 0.205 | A random model scores the base rate, 0.114, so the baseline ranks about 1.8x better than chance. |
| Recall at precision >= 0.2 | 0.357 | Flagging only where at least 1 in 5 flags is a real return, about 36% of readmissions are caught. |
| "Always no" accuracy | 0.886 | Shows why accuracy is not a headline metric: a model that catches nobody scores 89%. |

**Calibration caveat:** class weighting inflates the scores. In the validation set, patients scored about 0.88 were readmitted about 40% of the time. Use `risk_score` to rank patients, not as a literal probability.

## Reproduce

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 1. data: put diabetic_data.csv and IDS_mapping.csv in data/raw/ (UCI dataset 296)
# 2. pipeline
python -m src.clean
python -m src.split
python -m src.train                # writes models/model.joblib

# 3. tests
python -m pytest

# 4. service
python scripts/make_sample_request.py
docker compose up --build
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Prometheus: http://localhost:9090 (query `predict_requests_total`).

## API

- `POST /predict` with `{"features": {...}}` returns `risk_score`, `model_version` and `top_factors`. A missing or non-numeric field returns 422 naming the problem; an unseen category still scores.
- `GET /health` returns `{"status": "ok"}`, or 503 if the model file is missing.
- `GET /metrics` exposes `predict_requests_total{status}` and `predict_latency_seconds`.

## Rollback

Keep the previous image tag; to roll back, set the image tag in `docker-compose.yml` to the previous one and run `docker compose up -d`.

## Not in Pass 1

MLflow, XGBoost/LightGBM, SHAP, Fairlearn, Evidently, Grafana, audit log, model card, cloud deploy.
