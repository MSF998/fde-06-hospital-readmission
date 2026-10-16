from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Real quantities: "more" means more (days, counts of visits or procedures).
NUMERIC_COLS: list[str] = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Labels: no order between values. The *_id columns look like numbers but are
# categories (clean.py already turned them into strings).
CATEGORICAL_COLS: list[str] = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]

# Deliberately absent: readmit_30 and readmitted (the target), patient_nbr and
# encounter_id (identifiers, not predictors).


def build_preprocessor() -> ColumnTransformer:
    """Two transformers, named exactly "num" and "cat"."""
    # Numbers: fill blanks with the median, then rescale to mean 0 / spread 1.
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # Categories: fill blanks with "Missing", then one 0/1 column per value.
    # handle_unknown="ignore": a value never seen in training becomes all zeros
    # instead of crashing the live service.
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # (name, recipe, columns). The names "num" and "cat" are used by the tests
    # and the API, so keep them exact.
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
        ]
    )
