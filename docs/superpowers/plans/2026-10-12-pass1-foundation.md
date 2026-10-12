# Pass 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thin end-to-end slice: raw diabetes data -> cleaned data -> patient-level split -> baseline logistic regression -> FastAPI service in Docker Compose with Prometheus, so `curl /predict` returns a risk score.

**Architecture:** Four small pipeline scripts in `src/` (clean, split, features, train) produce one sklearn `Pipeline` saved with joblib. A FastAPI app in `api/` loads that file at startup and serves `/predict`, `/health` and `/metrics`. Docker Compose runs the API next to Prometheus.

**Tech Stack:** Python 3.13, pandas, scikit-learn, pyarrow, FastAPI, Uvicorn, prometheus-client, pytest, Docker Compose, Prometheus.

**Spec:** `docs/superpowers/specs/2026-10-12-pass1-foundation-design.md`

## Who does what (this is a learning project)

The project owner is learning. Tasks 2-4 (`src/clean.py`, `src/split.py`, `src/features.py`, `src/train.py`) are **owner-written**. For those tasks the plan gives the tests in full, the exact function signatures, and hints, but deliberately **no solution code**. An agent executing the plan must **not** write those implementations; instead it explains the concept, lets the owner write the code, runs the tests, and reviews the result. Tasks 1 and 5-7 (setup, API, Docker, Prometheus, README) are **Claude-written** and contain full code. Steps marked **[OWNER]** are the owner's; steps marked **[CLAUDE]** are Claude's.

## Global Constraints

- Local only: no cloud deploy, no cloud accounts, no secrets or keys in the repo (`.env` is git-ignored).
- Target: `readmitted == "<30"` -> 1, everything else -> 0; the column is named `readmit_30`.
- Split by `patient_nbr` into train/val/test of about 70/15/15; the test set is not used in Pass 1 (touched once, at the very end of the project).
- Fit encoders and scalers on train only; preprocessing and model live in one sklearn `Pipeline`.
- Baseline model: logistic regression with `class_weight="balanced"`.
- Headline metrics: PR-AUC and recall at a fixed precision, plus a calibration check. Accuracy is not a headline metric.
- `weight` column is dropped. A missing A1c is never filled with a normal value.
- API: `POST /predict`, `GET /health`, `GET /metrics`; bad input returns 422.
- `python -m src.train` rebuilds the model from raw data; `docker compose up` starts API + Prometheus.
- The pickled pipeline may use only scikit-learn classes (no lambdas, no functions from `src/`), so it loads inside the container.
- Pass 1 does not include: MLflow, XGBoost/LightGBM, SHAP, Fairlearn, Evidently, Grafana, audit log, model card, DVC, cloud deploy.

## Review Focus

Inputs and conditions the spec implies but does not spell out, most likely first. Each has a test in the task that owns the code.

- A categorical field sent as `null` (JSON) must be treated as missing, not crash or become the string "None". (Task 5)
- A categorical value never seen in training (e.g. `race: "Martian"`) must still return a score, not a 500. (Tasks 4 and 5)
- A request missing a required feature must return 422 naming the missing fields. (Task 5)
- A non-numeric value in a numeric field (e.g. `"time_in_hospital": "abc"`) must return 422, not 500. (Task 5)
- The raw CSV uses the string `None` for "not measured" (A1c, glucose); pandas' default NA handling silently turns it into NaN. Reading must be explicit about this. (Task 2)
- The model file is missing at API startup: the service must start, `/health` and `/predict` return 503 with a clear message. (Task 5)

## File Structure

| File | Responsibility | Owner |
| --- | --- | --- |
| `requirements.txt` | Pinned dependencies for local and Docker | Claude |
| `.gitignore` (modify) | Ignore data, models, venv, sample request | Claude |
| `docs/decisions.md` | Log of cleaning and feature decisions with reasons | Owner |
| `src/__init__.py`, `api/__init__.py`, `tests/__init__.py` | Make packages importable | Claude |
| `src/clean.py` | Load raw CSV, clean, define target, write `data/processed/clean.parquet` | Owner |
| `src/split.py` | Patient-level train/val/test split, writes three parquet files | Owner |
| `src/features.py` | Column lists and `build_preprocessor()` | Owner |
| `src/train.py` | Metrics helpers, `build_pipeline()`, training entry point | Owner |
| `api/main.py` | FastAPI app, Prometheus metrics, scoring | Claude |
| `tests/conftest.py` | Tiny test model and API client fixtures | Claude |
| `tests/test_clean.py`, `tests/test_split.py`, `tests/test_train.py` | Pipeline tests | Claude writes, owner implements to pass |
| `tests/test_api.py` | API tests | Claude |
| `scripts/make_sample_request.py` | Builds `sample_request.json` from real data | Claude |
| `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `monitoring/prometheus.yml` | Container and monitoring stack | Claude |
| `README.md` (modify) | Reproduce everything from raw data to running service | Claude |

---

### Task 1: Project setup, dependencies and raw data

**Files:**
- Create: `requirements.txt`, `src/__init__.py`, `api/__init__.py`, `tests/__init__.py`, `docs/decisions.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: a working virtualenv with all dependencies; `data/raw/diabetic_data.csv` on disk; `docs/decisions.md` for the owner's notes.

- [ ] **Step 1: [CLAUDE] Create `requirements.txt`**

```text
numpy==2.2.2
pandas==2.2.3
pyarrow==19.0.0
scikit-learn==1.6.1
scipy==1.15.1
joblib==1.4.2
fastapi==0.115.8
uvicorn==0.34.0
pydantic==2.10.6
prometheus-client==0.21.1
pytest==8.3.4
httpx==0.28.1
```

If a pin fails to install on this machine, bump it to the nearest working version and note the change in the commit message.

- [ ] **Step 2: [CLAUDE] Append to `.gitignore`**

```text
# project
data/raw/
data/processed/
models/
sample_request.json
.venv/
.env
```

- [ ] **Step 3: [CLAUDE] Create package markers and the decisions log**

Create empty files `src/__init__.py`, `api/__init__.py`, `tests/__init__.py`. Create `docs/decisions.md`:

```markdown
# Decisions log

Owner's notes: for every cleaning and feature decision, write what you chose and why.

## Cleaning
(fill in as you go)

## Features
(fill in as you go)
```

- [ ] **Step 4: [CLAUDE] Create the virtualenv and install**

Run (Git Bash, from the repo root):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Expected: install completes with no errors.

- [ ] **Step 5: [CLAUDE] Get the raw data**

Download the UCI zip, which contains `diabetic_data.csv` and `IDS_mapping.csv`:

```bash
mkdir -p data/raw
curl -L -o data/raw/diabetes.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
cd data/raw && unzip -o diabetes.zip && cd ../..
ls data/raw
```

Expected: `diabetic_data.csv` is listed. If the URL fails, download the dataset from https://archive.ics.uci.edu/dataset/296 (or the Kaggle mirror) in a browser and put `diabetic_data.csv` in `data/raw/`.

- [ ] **Step 6: [OWNER] Profile the raw data**

In a scratch notebook under `notebooks/` or a Python shell, answer these and write the answers into `docs/decisions.md` under "Cleaning":

1. How many rows and columns? What is the share of each `readmitted` value (`<30`, `>30`, `NO`)?
2. Count the `"?"` values per column. Which columns are mostly missing?
3. How many distinct `patient_nbr` values are there versus rows?
4. What are the value counts of `discharge_disposition_id`? Look up the meaning of each code in `data/raw/IDS_mapping.csv`. Which codes mean the patient died or went to hospice?

Hint: `pd.read_csv(path)` then `df.isin(["?"]).sum()`, `df["readmitted"].value_counts(normalize=True)`, `df["patient_nbr"].nunique()`.

- [ ] **Step 7: [CLAUDE] Commit**

```bash
git add requirements.txt .gitignore src/__init__.py api/__init__.py tests/__init__.py docs/decisions.md
git commit -m "chore: project setup, dependencies and decisions log"
```

---

### Task 2: Cleaning (`src/clean.py`) - owner-written

**Files:**
- Create: `src/clean.py` (owner), `tests/test_clean.py` (Claude)

**Interfaces:**
- Consumes: `data/raw/diabetic_data.csv`.
- Produces:
  - `load_raw(path: str) -> pd.DataFrame` - reads the CSV; `"?"` becomes NaN; the string `"None"` is kept as a normal value.
  - `clean(df: pd.DataFrame) -> pd.DataFrame` - returns the cleaned frame with a `readmit_30` column (int 0/1).
  - `main() -> None` - reads `data/raw/diabetic_data.csv`, writes `data/processed/clean.parquet`; run with `python -m src.clean`.

**Concept to explain first (Claude):** why `"?"` must become NaN, and why pandas' default `na_values` list (which includes the string `"None"` in pandas 2.x) would silently turn "not measured" into NaN. Show `keep_default_na=False`.

- [ ] **Step 1: [CLAUDE] Write the failing tests** in `tests/test_clean.py`

```python
import numpy as np
import pandas as pd

from src.clean import clean, load_raw


def make_raw_frame():
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "patient_nbr": [10, 11, 12, 13, 14],
            "race": ["Caucasian", "?", "AfricanAmerican", "Caucasian", "Asian"],
            "gender": ["Male", "Female", "Female", "Male", "Female"],
            "age": ["[70-80)", "[50-60)", "[60-70)", "[40-50)", "[80-90)"],
            "weight": ["?", "?", "?", "?", "?"],
            "admission_type_id": [1, 2, 1, 3, 1],
            "discharge_disposition_id": [1, 11, 1, 13, 1],
            "admission_source_id": [7, 1, 7, 1, 7],
            "time_in_hospital": [3, 5, 2, 7, 4],
            "payer_code": ["?", "MC", "?", "SP", "MC"],
            "medical_specialty": ["?", "?", "Cardiology", "?", "Surgery"],
            "num_medications": [10, 20, 5, 15, 8],
            "A1Cresult": ["None", ">8", "None", "Norm", "None"],
            "readmitted": ["<30", ">30", "NO", "<30", "NO"],
        }
    )


def test_load_raw_turns_question_mark_into_nan_but_keeps_none_string(tmp_path):
    path = tmp_path / "raw.csv"
    make_raw_frame().to_csv(path, index=False)
    df = load_raw(str(path))
    assert df["race"].isna().sum() == 1
    assert not (df == "?").any().any()
    assert (df["A1Cresult"] == "None").sum() == 3


def test_clean_drops_weight_column():
    df = clean(load_from_frame(make_raw_frame()))
    assert "weight" not in df.columns


def test_clean_target_is_one_only_for_less_than_30():
    raw = make_raw_frame()
    df = clean(load_from_frame(raw))
    by_id = df.set_index("encounter_id")["readmit_30"]
    assert by_id[1] == 1
    assert by_id[3] == 0
    assert set(df["readmit_30"].unique()) <= {0, 1}


def test_clean_removes_death_and_hospice_discharges():
    df = clean(load_from_frame(make_raw_frame()))
    assert not df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21]).any()
    assert sorted(df["encounter_id"]) == [1, 3, 5]


def test_clean_leaves_no_missing_values_in_payer_and_specialty():
    df = clean(load_from_frame(make_raw_frame()))
    for col in ["payer_code", "medical_specialty"]:
        assert col not in df.columns or df[col].notna().all()


def test_clean_keeps_not_measured_a1c_as_its_own_value():
    df = clean(load_from_frame(make_raw_frame()))
    assert df["A1Cresult"].notna().all()
    assert (df["A1Cresult"] == "None").any() or (df["A1Cresult"] == "Not measured").any()
    assert not df["A1Cresult"].isin(["Norm"]).all()


def test_clean_casts_id_columns_to_strings():
    df = clean(load_from_frame(make_raw_frame()))
    for col in ["admission_type_id", "admission_source_id"]:
        assert df[col].map(type).eq(str).all()


def test_clean_does_not_mutate_its_input():
    raw = load_from_frame(make_raw_frame())
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def load_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Mimic load_raw's output without touching disk."""
    return frame.replace("?", np.nan)
```

- [ ] **Step 2: [CLAUDE] Run tests to verify they fail**

Run: `python -m pytest tests/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.clean'` (or ImportError).

- [ ] **Step 3: [OWNER] Write `src/clean.py`**

Create the file with these exact signatures and make the tests pass:

```python
import pandas as pd

RAW_PATH = "data/raw/diabetic_data.csv"
OUT_PATH = "data/processed/clean.parquet"


def load_raw(path: str) -> pd.DataFrame:
    """Read the raw CSV. "?" -> NaN. The string "None" must stay a normal value."""
    ...


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with a `readmit_30` column. Do not modify `df`."""
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

Replace each `...` with your own code. Hints:
- `pd.read_csv(path, na_values=["?"], keep_default_na=False)` is the read you want; try it without `keep_default_na=False` and see what happens to `A1Cresult`.
- `df.drop(columns=[...])`, `df.copy()`, `df[col].astype(str)`, `df["readmitted"].eq("<30").astype(int)`.
- Discharge codes for death and hospice in this dataset: 11, 13, 14, 19, 20, 21 (confirm against `IDS_mapping.csv` and your notes in `docs/decisions.md`).
- For `payer_code` and `medical_specialty`, choose: drop them, or `fillna("Missing")`. Record your choice and reason in `docs/decisions.md`.
- Also fill missing `race`, `diag_1`, `diag_2`, `diag_3` sensibly (for example `"Missing"`) and note it. `main()` should create `data/processed/` if needed (`os.makedirs(..., exist_ok=True)`) and write with `df.to_parquet(OUT_PATH, index=False)`.

- [ ] **Step 4: [CLAUDE] Run tests to verify they pass**

Run: `python -m pytest tests/test_clean.py -v`
Expected: all PASS. If a test fails, Claude explains what it checks and points to the concept; the owner fixes the code.

- [ ] **Step 5: [OWNER] Run the script on the real data and sanity-check**

Run: `python -m src.clean`
Then in a shell: load `data/processed/clean.parquet` and confirm: row count is a bit under 101,766; `readmit_30.mean()` is about 0.11 (slightly higher after removing the dead and hospice patients); no `"?"` remains.

- [ ] **Step 6: [CLAUDE] Review and commit**

Claude reviews `src/clean.py` for leakage risks (for example, does anything use `readmitted` as a feature?) and for the decisions logged in `docs/decisions.md`.

```bash
git add src/clean.py tests/test_clean.py docs/decisions.md
git commit -m "feat: cleaning pipeline with target definition"
```

---

### Task 3: Patient-level split (`src/split.py`) - owner-written

**Files:**
- Create: `src/split.py` (owner), `tests/test_split.py` (Claude)

**Interfaces:**
- Consumes: `data/processed/clean.parquet` with columns `patient_nbr` and `readmit_30`.
- Produces:
  - `split_by_patient(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` returning `(train, val, test)`.
  - `main() -> None` writing `data/processed/train.parquet`, `val.parquet`, `test.parquet`; run with `python -m src.split`.

**Concept to explain first (Claude):** leakage. One patient can have many encounters; a random row split puts the same patient in train and test, so the model "remembers" patients and scores look better than reality. Group-aware splitting fixes this (`GroupShuffleSplit`).

- [ ] **Step 1: [CLAUDE] Write the failing tests** in `tests/test_split.py`

```python
import numpy as np
import pandas as pd

from src.split import split_by_patient


def make_frame(n_patients=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        for _ in range(rng.integers(1, 6)):
            rows.append({"patient_nbr": pid, "readmit_30": int(rng.random() < 0.11), "x": rng.random()})
    return pd.DataFrame(rows)


def test_no_patient_appears_in_more_than_one_set():
    train, val, test = split_by_patient(make_frame())
    p_train, p_val, p_test = (set(d["patient_nbr"]) for d in (train, val, test))
    assert p_train.isdisjoint(p_val)
    assert p_train.isdisjoint(p_test)
    assert p_val.isdisjoint(p_test)


def test_every_row_lands_in_exactly_one_set():
    df = make_frame()
    train, val, test = split_by_patient(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_set_sizes_are_close_to_70_15_15():
    df = make_frame()
    train, val, test = split_by_patient(df)
    n = len(df)
    assert abs(len(train) / n - 0.70) < 0.04
    assert abs(len(val) / n - 0.15) < 0.04
    assert abs(len(test) / n - 0.15) < 0.04


def test_split_is_deterministic_for_a_given_seed():
    df = make_frame()
    a = split_by_patient(df, seed=7)
    b = split_by_patient(df, seed=7)
    for x, y in zip(a, b):
        assert sorted(x["patient_nbr"].unique()) == sorted(y["patient_nbr"].unique())


def test_positive_rate_is_similar_across_sets():
    train, val, test = split_by_patient(make_frame())
    rates = [d["readmit_30"].mean() for d in (train, val, test)]
    assert max(rates) - min(rates) < 0.05
```

- [ ] **Step 2: [CLAUDE] Run tests to verify they fail**

Run: `python -m pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.split'`.

- [ ] **Step 3: [OWNER] Write `src/split.py`**

```python
import pandas as pd

IN_PATH = "data/processed/clean.parquet"
OUT_DIR = "data/processed"


def split_by_patient(
    df: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test). No `patient_nbr` may appear in more than one set."""
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

Replace each `...` with your own code. Hints:
- `sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=..., random_state=seed)` with `groups=df["patient_nbr"]`. Use it twice: first carve out the test set, then split the remainder into train and val. Careful: the second `test_size` is a fraction of what is left, not of the whole.
- `.split(df, groups=...)` returns index arrays; use `df.iloc[idx]`.
- `main()` reads `IN_PATH`, splits, prints the size and positive rate of each set, and writes `train.parquet`, `val.parquet`, `test.parquet` into `OUT_DIR`.

- [ ] **Step 4: [CLAUDE] Run tests to verify they pass**

Run: `python -m pytest tests/test_split.py -v`
Expected: all PASS.

- [ ] **Step 5: [OWNER] Run on the real data**

Run: `python -m src.split`
Expected: three files in `data/processed/`; the printed positive rate is close to 11% in each set. Write one sentence in `docs/decisions.md` on why the rates are (or are not) similar.

- [ ] **Step 6: [CLAUDE] Review and commit**

```bash
git add src/split.py tests/test_split.py docs/decisions.md
git commit -m "feat: patient-level train/val/test split"
```

---

### Task 4: Features and baseline model (`src/features.py`, `src/train.py`) - owner-written

**Files:**
- Create: `src/features.py` (owner), `src/train.py` (owner), `tests/test_train.py` (Claude)

**Interfaces:**
- Consumes: `data/processed/train.parquet` and `val.parquet` (columns from Task 2, including `readmit_30`).
- Produces (`src/features.py`):
  - `NUMERIC_COLS: list[str]`, `CATEGORICAL_COLS: list[str]`
  - `build_preprocessor() -> sklearn.compose.ColumnTransformer` with exactly two transformers named `"num"` and `"cat"`.
- Produces (`src/train.py`):
  - `build_pipeline() -> sklearn.pipeline.Pipeline` with steps named `"prep"` (the preprocessor) and `"clf"` (a `LogisticRegression` with `class_weight="balanced"`).
  - `recall_at_precision(y_true, scores, min_precision: float) -> float`
  - `always_no_accuracy(y_true) -> float`
  - `evaluate(y_true, scores, min_precision: float = 0.2) -> dict` with keys `"pr_auc"`, `"recall_at_precision"`, `"brier_score"`, `"min_precision"`.
  - `calibration_bins(y_true, scores, n_bins: int = 5) -> pd.DataFrame` with columns `"mean_predicted"`, `"observed_rate"`.
  - `main() -> None`; run with `python -m src.train`; writes `models/model.joblib`.
- The API (Task 5) relies on: step names `"prep"`/`"clf"`, transformer names `"num"`/`"cat"`, `pipeline.feature_names_in_`, `clf.coef_`, and `prep.get_feature_names_out()`.

**Concepts to explain first (Claude):** (1) why fit encoders and scalers on train only, and how a `Pipeline` guarantees it; (2) why class weighting helps when only 11% are positive; (3) accuracy versus PR-AUC, using the "always no" model (about 89% accurate, catches nobody); (4) precision, recall and what "recall at a fixed precision" means for a care team with limited follow-up capacity; (5) calibration.

- [ ] **Step 1: [CLAUDE] Write the failing tests** in `tests/test_train.py`

```python
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
```

- [ ] **Step 2: [CLAUDE] Run tests to verify they fail**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 3: [OWNER] Write `src/features.py`**

```python
from sklearn.compose import ColumnTransformer

NUMERIC_COLS: list[str] = [...]
CATEGORICAL_COLS: list[str] = [...]


def build_preprocessor() -> ColumnTransformer:
    """Two transformers, named exactly "num" and "cat"."""
    ...
```

Replace each `...` with your own code. Hints:
- Start with numerics `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`, and categoricals `race`, `gender`, `age`, `admission_type_id`, `admission_source_id`, `max_glu_serum`, `A1Cresult`, `change`, `diabetesMed`. Change the lists if you have a reason, and write the reason in `docs/decisions.md` under "Features" (one line per feature: why it exists).
- `num`: `Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])`.
- `cat`: `Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])`. Think about why `handle_unknown="ignore"` matters for a live service.
- Use only scikit-learn classes in here; no `lambda` and no helper functions (the test pickles the pipeline and checks that it does not point back to `src`).

- [ ] **Step 4: [OWNER] Write `src/train.py`**

```python
import pandas as pd

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
MODEL_PATH = "models/model.joblib"


def build_pipeline():
    """Pipeline with steps named "prep" and "clf"."""
    ...


def recall_at_precision(y_true, scores, min_precision: float) -> float:
    """Highest recall among thresholds whose precision >= min_precision; 0.0 if none."""
    ...


def always_no_accuracy(y_true) -> float:
    """Accuracy of a model that predicts 0 for everyone."""
    ...


def calibration_bins(y_true, scores, n_bins: int = 5) -> pd.DataFrame:
    """Columns: mean_predicted, observed_rate."""
    ...


def evaluate(y_true, scores, min_precision: float = 0.2) -> dict:
    """Keys: pr_auc, recall_at_precision, brier_score, min_precision."""
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

Replace each `...` with your own code. Hints:
- `sklearn.metrics.precision_recall_curve`, `average_precision_score` (this is the standard PR-AUC), `brier_score_loss`; `sklearn.calibration.calibration_curve(y_true, scores, n_bins=n_bins)` returns `(observed, predicted)`.
- `precision_recall_curve` returns precision and recall arrays one element longer than thresholds; the last point (precision 1, recall 0) is a placeholder. Mask on `precision >= min_precision`, then take the max recall; return `0.0` if the mask is empty.
- `main()`: read train and val parquet, build `X = df[NUMERIC_COLS + CATEGORICAL_COLS]` and `y = df["readmit_30"]`, fit the pipeline on train only, score validation with `predict_proba(...)[:, 1]`, print `evaluate(...)`, the `calibration_bins` table and `always_no_accuracy(y_val)`, then `os.makedirs("models", exist_ok=True)` and `joblib.dump(pipe, MODEL_PATH)`. Do not load `test.parquet`.
- Choose `min_precision` deliberately: the base rate is about 11%, so 0.2 means "flag patients where at least 1 in 5 really returns". Write your reasoning in `docs/decisions.md`.

- [ ] **Step 5: [CLAUDE] Run tests to verify they pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: all PASS.

- [ ] **Step 6: [OWNER] Train on the real data**

Run: `python -m src.train`
Expected: prints metrics and a calibration table, and `models/model.joblib` exists. Write down: PR-AUC (compare it with the base rate of about 0.11, which is what a random model scores), recall at your chosen precision, and the "always no" accuracy. In `docs/decisions.md`, one paragraph: does the baseline beat random, and how can you tell?

- [ ] **Step 7: [CLAUDE] Review and commit**

Claude checks that nothing in `train.py` touches the test set, and that no encoder is fitted outside the pipeline.

```bash
git add src/features.py src/train.py tests/test_train.py docs/decisions.md
git commit -m "feat: baseline logistic regression pipeline and metrics"
```

---

### Task 5: FastAPI service (`api/main.py`) - Claude-written

**Files:**
- Create: `api/main.py`, `tests/conftest.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: a joblib-pickled `Pipeline` with steps `"prep"` (a `ColumnTransformer` with transformers named `"num"` and `"cat"`) and `"clf"` (a fitted `LogisticRegression`), from the path in env var `MODEL_PATH` (default `models/model.joblib`). Version label from env `MODEL_VERSION` (default `baseline-lr-v1`).
- Produces: the ASGI app `api.main:app`:
  - `POST /predict`, body `{"features": {<column>: <value>, ...}}` -> `{"risk_score": float, "model_version": str, "top_factors": [{"feature": str, "contribution": float}, ...]}`
  - `GET /health` -> `{"status": "ok"}` (503 if no model)
  - `GET /metrics` -> Prometheus text; metrics `predict_requests_total{status}` (counter) and `predict_latency_seconds` (histogram).

- [ ] **Step 1: [CLAUDE] Write the test fixtures** in `tests/conftest.py`

These build a tiny model with the same structure as the real pipeline, so API tests do not need the real data.

```python
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
```

- [ ] **Step 2: [CLAUDE] Write the failing tests** in `tests/test_api.py`

```python
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_score_version_and_factors(client):
    r = client.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_version"] == "baseline-lr-v1"
    assert len(body["top_factors"]) >= 1
    assert set(body["top_factors"][0]) == {"feature", "contribution"}


def test_predict_unseen_category_still_scores(client):
    r = client.post("/predict", json={"features": {"n1": 0.5, "c1": "Martian"}})
    assert r.status_code == 200


def test_predict_null_category_is_treated_as_missing(client):
    r = client.post("/predict", json={"features": {"n1": 0.5, "c1": None}})
    assert r.status_code == 200


def test_predict_integer_category_matches_string_category(client):
    a = client.post("/predict", json={"features": {"n1": 0.5, "c1": 2}}).json()["risk_score"]
    b = client.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}}).json()["risk_score"]
    assert a == b


def test_predict_missing_feature_returns_422_naming_the_field(client):
    r = client.post("/predict", json={"features": {"n1": 0.5}})
    assert r.status_code == 422
    assert "c1" in r.text


def test_predict_non_numeric_value_returns_422(client):
    r = client.post("/predict", json={"features": {"n1": "abc", "c1": "2"}})
    assert r.status_code == 422


def test_predict_malformed_body_returns_422(client):
    r = client.post("/predict", json={"nope": 1})
    assert r.status_code == 422


def test_metrics_exposes_counter_and_histogram(client):
    client.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}})
    text = client.get("/metrics").text
    assert "predict_requests_total" in text
    assert "predict_latency_seconds" in text


def test_service_without_model_starts_and_reports_503(client_no_model):
    assert client_no_model.get("/health").status_code == 503
    r = client_no_model.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}})
    assert r.status_code == 503
```

- [ ] **Step 3: [CLAUDE] Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'`.

- [ ] **Step 4: [CLAUDE] Write `api/main.py`**

```python
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
```

- [ ] **Step 5: [CLAUDE] Run tests to verify they pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: all PASS. If `test_predict_non_numeric_value_returns_422` returns 500, the imputer raised an exception type not caught above; catch that type too and re-run.

- [ ] **Step 6: [CLAUDE] Commit**

```bash
git add api/main.py tests/conftest.py tests/test_api.py
git commit -m "feat: FastAPI scoring service with Prometheus metrics"
```

---

### Task 6: Docker, Compose and Prometheus - Claude-written

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `monitoring/prometheus.yml`, `scripts/make_sample_request.py`

**Interfaces:**
- Consumes: `models/model.joblib` (from Task 4), `requirements.txt`, `api/main.py`.
- Produces: `docker compose up` serves the API on `localhost:8000` and Prometheus on `localhost:9090`; `sample_request.json` for manual testing.

- [ ] **Step 1: [CLAUDE] Create `Dockerfile`**

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY models/model.joblib models/model.joblib

ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: [CLAUDE] Create `.dockerignore`**

```text
.venv/
data/
notebooks/
docs/
tests/
.git/
__pycache__/
sample_request.json
```

Note: `models/` is not ignored because the Dockerfile copies `models/model.joblib`.

- [ ] **Step 3: [CLAUDE] Create `monitoring/prometheus.yml`**

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: readmission-api
    static_configs:
      - targets: ["api:8000"]
```

- [ ] **Step 4: [CLAUDE] Create `docker-compose.yml`**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      MODEL_VERSION: baseline-lr-v1

  prometheus:
    image: prom/prometheus:v2.55.1
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - api
```

- [ ] **Step 5: [CLAUDE] Create `scripts/make_sample_request.py`**

```python
"""Write sample_request.json from the first training row, for manual curl tests."""
import json

import joblib
import pandas as pd

model = joblib.load("models/model.joblib")
row = pd.read_parquet("data/processed/train.parquet").iloc[0]
features = {}
for col in model.feature_names_in_:
    value = row[col]
    features[col] = None if pd.isna(value) else (value.item() if hasattr(value, "item") else value)

with open("sample_request.json", "w") as f:
    json.dump({"features": features}, f, indent=2)
print("wrote sample_request.json")
```

- [ ] **Step 6: [CLAUDE] Build and start the stack**

Run: `python scripts/make_sample_request.py && docker compose up --build -d`
Expected: image builds, both containers start. Check with `docker compose ps`.

- [ ] **Step 7: [CLAUDE] Verify the endpoints**

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
curl -s "localhost:9090/api/v1/query?query=predict_requests_total"
```

Expected: `{"status":"ok"}`; a JSON response with `risk_score` between 0 and 1, `model_version` and `top_factors`; a Prometheus response containing `predict_requests_total` with value >= 1 (wait about 10 seconds after the predict call for a scrape). Send the predict request three more times and re-query: the value increases.

- [ ] **Step 8: [CLAUDE] Stop the stack and commit**

```bash
docker compose down
git add Dockerfile .dockerignore docker-compose.yml monitoring/prometheus.yml scripts/make_sample_request.py
git commit -m "feat: Dockerized API with Prometheus via Compose"
```

---

### Task 7: README and end-to-end check - Claude-written

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a README that reproduces Pass 1 from raw data to running service.

- [ ] **Step 1: [CLAUDE] Replace `README.md` content**

````markdown
# 30-day hospital readmission risk (Pass 1)

Predict, at discharge, whether a diabetic patient will be readmitted within 30 days. Pass 1 is a thin end-to-end slice: data -> baseline model -> API -> Docker + Prometheus. See `docs/superpowers/specs/` for the design and `docs/decisions.md` for the choices made and why.

## Reproduce

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 1. data: put diabetic_data.csv in data/raw/ (UCI dataset 296)
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

## Rollback

Keep the previous image tag; to roll back, set the image tag in `docker-compose.yml` to the previous one and run `docker compose up -d`.

## Not in Pass 1

MLflow, XGBoost/LightGBM, SHAP, Fairlearn, Evidently, Grafana, audit log, model card, cloud deploy.
````

- [ ] **Step 2: [CLAUDE] Run the whole test suite**

Run: `python -m pytest -v`
Expected: all tests in `tests/` PASS.

- [ ] **Step 3: [CLAUDE] Fresh-clone check of the "done" definition**

Delete `data/processed/` and `models/`, then run, in order: `python -m src.clean`, `python -m src.split`, `python -m src.train`, `python scripts/make_sample_request.py`, `docker compose up --build -d`, the `curl` from the README, and the Prometheus query. Expected: every step works with no manual fixes; `docker compose down` afterwards.

- [ ] **Step 4: [CLAUDE] Commit**

```bash
git add README.md
git commit -m "docs: README with reproduction steps for Pass 1"
```

---

## Self-review

- **Spec coverage:** repo layout (Task 1, file structure); clean, split, features, train (Tasks 2-4, owner); API with `/predict`, `/health`, `/metrics` and top factors (Task 5); Docker, Compose, Prometheus (Task 6); pytest checks for health, valid and malformed predict, and split leakage (Tasks 3, 5); definition of done, 4 items (Tasks 4, 6, 7). Open decisions from the spec (payer/specialty, discharge codes, feature set, precision level) are assigned to the owner in Tasks 2 and 4 with a place to record them in `docs/decisions.md`. Rollback note appears in the README.
- **Placeholders:** the only `...` in this plan are the owner's deliberate stubs in Tasks 2-4, which the tests define; everything Claude writes is shown in full.
- **Type consistency:** `readmit_30`, `patient_nbr`, `NUMERIC_COLS`, `CATEGORICAL_COLS`, step names `prep`/`clf`, transformer names `num`/`cat`, and `models/model.joblib` are used identically across tasks.
