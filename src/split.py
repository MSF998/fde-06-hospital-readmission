import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

IN_PATH = "data/processed/clean.parquet"
OUT_DIR = "data/processed"


def split_by_patient(
    df: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test). No `patient_nbr` may appear in more than one set."""
    # Split 1: carve out the test set, keeping each patient's rows together.
    first = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    rest_idx, test_idx = next(first.split(df, groups=df["patient_nbr"]))
    rest = df.iloc[rest_idx]
    test = df.iloc[test_idx]

    # Split 2: val is val_frac of the WHOLE data, so a larger fraction of `rest`.
    val_size = val_frac / (1 - test_frac)
    second = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(second.split(rest, groups=rest["patient_nbr"]))
    train = rest.iloc[train_idx]
    val = rest.iloc[val_idx]

    return train, val, test


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    train, val, test = split_by_patient(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        # print(name, len(part), round(part["readmit_30"].mean(), 4))
        part.to_parquet(f"{OUT_DIR}/{name}.parquet", index=False)


if __name__ == "__main__":
    main()
