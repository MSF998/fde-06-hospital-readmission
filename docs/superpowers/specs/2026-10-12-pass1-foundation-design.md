# Pass 1 Foundation: Design Spec

Date: 2026-10-12
Project: 30-day hospital readmission risk (see `assignment.pdf`)

## Purpose and intent

The assignment asks for a complete, production-style ML system: data pipeline, model, deployed service, monitoring and governance. Modeling is about 20% of the grade; the system around it is about 80%.

This project is for **learning**. The goal is that the author can explain and defend every decision (why accuracy misleads, what leakage is, what drift means), not just produce a passing system. A messy but working foundation is acceptable; it will be improved afterwards.

**Scope decisions (agreed with the author):**
- Local only. No cloud deploy for now; the cloud step stays as a documented path to run later.
- Thin vertical slice first (Pass 1), then deepen each stage (Pass 2). Pass 2 gets its own specs.
- Roles: the author writes the core ML logic (cleaning, split, features, training, metric choices). Claude writes boilerplate (API, Docker, compose, Prometheus config, README skeleton) and reviews the author's code. Claude explains each concept before the author uses it.

Reference: the requirements explainer doc is at https://claude.ai/code/artifact/61b162d0-3cb5-41ad-ba01-465db26e2521

## Pass 1 definition of done

1. `python -m src.train` rebuilds the model from the raw data.
2. `docker compose up` starts the API and Prometheus.
3. A `curl` request to `/predict` returns a risk score between 0 and 1.
4. Prometheus shows the request count increasing.

## Out of scope for Pass 1 (deferred to Pass 2)

Feature engineering beyond minimal encoding (ICD-9 grouping, engineered counts), XGBoost/LightGBM and tuning, MLflow, SHAP, Fairlearn, Evidently drift reports, Grafana, alerting and retraining trigger, audit log, model card, DVC, Great Expectations, cloud deploy, rollback documentation, and the 1-2 page reflection.

## Repository layout

```
data/raw/            # downloaded CSV (git-ignored)
data/processed/      # cleaned output (git-ignored)
src/
  clean.py           # "?" -> missing, drop junk columns, define target
  split.py           # split by patient_nbr into train/val/test
  features.py        # minimal encoding for Pass 1
  train.py           # baseline logistic regression, saves model.joblib
api/
  main.py            # FastAPI: /predict, /health, /metrics
tests/               # pytest
Dockerfile
docker-compose.yml   # api + prometheus
requirements.txt
README.md            # grows as we go
notebooks/           # exploration only; nothing depends on it
```

Ownership: the author writes `src/*`. Claude writes `api/`, Docker, compose, Prometheus config, `requirements.txt` and the README skeleton, and reviews `src/*`.

## Data and modeling flow

```
raw CSV -> clean.py -> processed data -> split.py -> train/val/test
                                              |
                          features.py -> train.py -> model.joblib
```

### Clean (`src/clean.py`)

- Read the CSV with `na_values="?"` so missing values are real NaNs from the start.
- Drop `weight` (about 97% empty). Decide on `payer_code` and `medical_specialty`: drop, or keep with "Missing" as an explicit category. Record the decision and reason.
- Target: `readmitted == "<30"` -> 1, everything else -> 0. Document this choice and why.
- A missing A1c becomes its own category ("not measured"); never fill with a normal value.
- Remove encounters that cannot be readmitted (patients who died or went to hospice), identified via `discharge_disposition_id`. The author inspects the column and decides which codes to remove. They would otherwise be counted as "not readmitted" and distort labels.
- Write the cleaned data to `data/processed/`.

### Split (`src/split.py`)

- Split by `patient_nbr` so no patient appears in more than one set.
- About 70/15/15 train/validation/test. The test set is touched once, at the end.
- Verify the positive rate is close to 11% in each set.

### Features (`src/features.py`, minimal in Pass 1)

- One-hot encode low-cardinality categoricals; keep numerics such as `number_inpatient`, `time_in_hospital`, `num_medications`.
- Fit all encoders/scalers on train only, then apply to validation and test.
- Bundle preprocessing and model into one sklearn Pipeline so the API applies identical steps.

### Train (`src/train.py`)

- Logistic regression with `class_weight="balanced"`.
- Report on validation: PR-AUC, recall at a fixed precision, and a simple calibration check.
- Compare against an "always no" model to show why accuracy misleads.
- Save the Pipeline with joblib to `models/model.joblib`.

## API, Docker and observability

### `api/main.py` (FastAPI)

- `POST /predict`: one patient's fields as JSON, validated with a Pydantic model (bad input returns 422). The model is loaded once at startup. Response: `{"risk_score": <float>, "model_version": "<string>", "top_factors": [...]}`.
- `GET /health`: liveness check.
- `GET /metrics`: Prometheus metrics.
- Top factors in Pass 1: for logistic regression, coefficient times feature value. A simple stub; replaced with SHAP in Pass 2.

### Metrics

Two Prometheus metrics: a request counter and a latency histogram.

### Docker and Compose

- Dockerfile: slim Python image, pinned dependencies from `requirements.txt`, copy `api/` and the saved model, run with uvicorn.
- `docker-compose.yml`: services `api` and `prometheus`; Prometheus scrapes `api:8000/metrics` every few seconds. Grafana is added in Pass 2.
- No secrets needed locally; configuration via environment variables, `.env` git-ignored.

## Testing

pytest, kept small:
- `/health` returns 200.
- A valid `/predict` returns a score in [0, 1].
- A malformed `/predict` input returns 422.
- `split.py`: no patient appears in more than one set (the leakage check).

## Concepts the author will meet in Pass 1

- Missing-data coding and why "?" must become NaN.
- Leakage, both split-by-patient and fit-on-train-only.
- Class imbalance, class weighting, and why accuracy misleads.
- PR-AUC, recall at fixed precision, calibration.
- What an sklearn Pipeline protects against.
- Containers, health checks and Prometheus scraping.

## Open decisions to settle while building

- What to do with `payer_code` and `medical_specialty`.
- Which `discharge_disposition_id` codes to exclude.
- Which numeric and categorical columns make the Pass 1 feature set.
- The precision level used for "recall at fixed precision".
