# Decisions log

Owner's notes: for every cleaning and feature decision, write what you chose and why.

## Cleaning

- How many rows and columns are there? What share is each readmitted value (<30, >30, NO)?
  - Columns: 50
  - Rows:101766
  - readmitted
    - NO 0.539119
    - > 30 0.349282
    - <30 0.111599

- How many "?" values does each column have? Which columns are mostly missing?
  - race: 2273
  - weight 98569
  - payer_code 40256
  - medical_specialty 49949
  - diag_1 21
  - diag_2 358
  - diag_3 1423

- How many distinct patient_nbr values are there compared with the number of rows?
  - 71518

- What are the value counts of discharge_disposition_id? Look the codes up in data/raw/IDS_mapping.csv. Which mean the patient died or went to hospice?
  - readmitted
    - NO 2337
    - <30 43
    - > 30 43

- dropped weight, because 97% of values are missing
- removed discharge codes 11, 13, 14, 19, 20 and 21, because those patients can't be readmitted
- filled payer*code, medical_specialty, race and diag*\* with "Missing" instead of dropping them, and say why to prevent ML models from crashing, avoid deleting valuable info
- the target is <30 = 1 and >30 or NO = 0
- the \*\_id columns are strings because they are categories

## Features

(fill in as you go)
