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


def load_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Mimic load_raw's output without touching disk."""
    return frame.replace("?", np.nan)


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
