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
