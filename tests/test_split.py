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
