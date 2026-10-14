import pandas as pd
import os

RAW_PATH = "data/raw/diabetic_data.csv"
OUT_PATH = "data/processed/clean.parquet"

DEAD_OR_HOSPICE = [11, 13, 14, 19, 20, 21]

def load_raw(path: str) -> pd.DataFrame:
    """Read the raw CSV. "?" -> NaN. The string "None" must stay a normal value."""
    return pd.read_csv(path, keep_default_na=False, na_values=['?'])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with a `readmit_30` column. Do not modify `df`."""
    df = df.copy()
    df = df.drop(columns=['weight'])
    df = df[~df['discharge_disposition_id'].isin(DEAD_OR_HOSPICE)]
    df['readmit_30'] = df['readmitted'].eq("<30").astype(int)
    for col in  ["race", "payer_code", "medical_specialty","diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")
    for col in ["admission_type_id", "admission_source_id","discharge_disposition_id"]:
        df[col] = df[col].astype(str)
    return df



def main() -> None:
    df = clean(load_raw(RAW_PATH))
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(df.shape, df["readmit_30"].mean())

if __name__ == "__main__":
    main()
