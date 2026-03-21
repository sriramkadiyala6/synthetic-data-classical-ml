# Check to see if the results from each batch are memorized or not by the LLM. 

import pandas as pd

# =============================
# LOAD DATA
# =============================
synthetic_path = "adult_synthetic_5000.csv"   # your generated data
real_path = "adult/adult.data"              # original dataset

synthetic = pd.read_csv(synthetic_path) #only for the first batch where we use headers

#synthetic = pd.read_csv(synthetic_path, header=None) #for the rest

# synthetic.columns = [
#     "age","workclass","fnlwgt","education","education-num",
#     "marital-status","occupation","relationship","race","sex",
#     "capital-gain","capital-loss","hours-per-week","native-country","income"
# ] # use when csv doesnt have a header column

real = pd.read_csv(real_path, header=None)

# Assign columns to real dataset
real.columns = synthetic.columns

print("===================================")
print("DATA VALIDATION REPORT")
print("===================================")

# =============================
# 1. EXACT MEMORIZATION CHECK
# =============================
merged = synthetic.merge(real.drop_duplicates(), how='inner')
print("\n[1] Exact matches with real data:", len(merged))

# =============================
# 2. DUPLICATE CHECK
# =============================
duplicates = synthetic.duplicated().sum()
print("[2] Duplicate rows in synthetic:", duplicates)

# =============================
# 3. MISSING / '?' CHECK
# =============================
missing = synthetic.isnull().sum().sum()
question_marks = synthetic.astype(str).apply(lambda x: x.str.contains(r"\?")).any(axis=1).sum()

print("[3] Missing values:", missing)
print("[3] Rows containing '?':", question_marks)

# =============================
# 4. NUMERICAL VALIDATION
# =============================
invalid_age = ((synthetic["age"] < 17) | (synthetic["age"] > 90)).sum()
invalid_hours = ((synthetic["hours-per-week"] < 1) | (synthetic["hours-per-week"] > 99)).sum()
invalid_gain = (synthetic["capital-gain"] < 0).sum()
invalid_loss = (synthetic["capital-loss"] < 0).sum()

print("\n[4] Invalid age rows:", invalid_age)
print("[4] Invalid hours-per-week rows:", invalid_hours)
print("[4] Invalid capital-gain rows:", invalid_gain)
print("[4] Invalid capital-loss rows:", invalid_loss)

# =============================
# 5. EDUCATION CONSISTENCY
# =============================
edu_map = {
    "HS-grad": 9,
    "Some-college": 10,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 16
}

def check_edu(row):
    if row["education"] in edu_map:
        return row["education-num"] == edu_map[row["education"]]
    return True

edu_mismatch = synthetic[~synthetic.apply(check_edu, axis=1)]
print("\n[5] Education mismatches:", len(edu_mismatch))

# =============================
# 6. DISTRIBUTION CHECKS
# =============================
print("\n[6] Income distribution:")
print(synthetic["income"].value_counts(normalize=True))

print("\n[6] Workclass distribution:")
print(synthetic["workclass"].value_counts(normalize=True).head())

print("\n[6] Hours-per-week (top values):")
print(synthetic["hours-per-week"].value_counts().head())

# =============================
# 7. LOGICAL SANITY CHECKS
# =============================
young_widowed = synthetic[(synthetic["age"] < 25) & (synthetic["marital-status"] == "Widowed")]
print("\n[7] Young widowed rows:", len(young_widowed))

# =============================
# FINAL SUMMARY
# =============================
print("\n===================================")
print("SUMMARY")
print("===================================")

print(f"Exact matches with real data: {len(merged)}")
print(f"Duplicate rows: {duplicates}")
print(f"Missing values: {missing}")
print(f"Rows with '?': {question_marks}")
print(f"Invalid age rows: {invalid_age}")
print(f"Invalid hours rows: {invalid_hours}")
print(f"Education mismatches: {len(edu_mismatch)}")

print("\nDONE.")