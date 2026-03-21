import pandas as pd

# =============================
# FILE PATHS
# =============================
synthetic_path = "credit_synthetic_combined.csv"
real_path = "default of credit card clients.xls"

# =============================
# LOAD DATA
# =============================
synthetic = pd.read_csv(synthetic_path)
real = pd.read_excel(real_path, header=1)

print("===================================")
print("DATA VALIDATION REPORT")
print("===================================")

# =============================
# 0. EXACT MEMORIZATION CHECK
# =============================
merged = synthetic.merge(real.drop_duplicates(), how='inner')
print("\n[0] Exact matches with real data:", len(merged))

# =============================
# 1. DUPLICATES
# =============================
duplicates = synthetic.duplicated().sum()
print("[1] Duplicate rows:", duplicates)

# =============================
# 2. MISSING / '?' CHECK
# =============================
missing = synthetic.isnull().sum().sum()
question_marks = synthetic.astype(str).apply(lambda x: x.str.contains(r"\?")).any(axis=1).sum()
print("[2] Missing values:", missing)
print("[2] Rows containing '?':", question_marks)

# =============================
# 3. NUMERICAL RANGE CHECKS
# =============================
invalid_age = ((synthetic["AGE"] < 21) | (synthetic["AGE"] > 79)).sum()
invalid_limit = (synthetic["LIMIT_BAL"] <= 0).sum()
print("\n[3] Invalid AGE rows:", invalid_age)
print("[3] Invalid LIMIT_BAL rows:", invalid_limit)

# =============================
# 4. CATEGORY VALIDATION
# =============================
invalid_sex = (~synthetic["SEX"].isin([1, 2])).sum()
invalid_edu = (~synthetic["EDUCATION"].isin([1, 2, 3, 4])).sum()
invalid_marriage = (~synthetic["MARRIAGE"].isin([1, 2, 3])).sum()
invalid_default = (~synthetic["default.payment.next.month"].isin([0, 1])).sum()
print("\n[4] Invalid SEX:", invalid_sex)
print("[4] Invalid EDUCATION:", invalid_edu)
print("[4] Invalid MARRIAGE:", invalid_marriage)
print("[4] Invalid default:", invalid_default)

# =============================
# 5. PAY_X VALIDATION
# =============================
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
invalid_pay = 0
for col in pay_cols:
    invalid_pay += (~synthetic[col].isin([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])).sum()
print("\n[5] Invalid PAY_X values:", invalid_pay)

# =============================
# 6. LOGICAL CHECKS
# =============================
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

pay_gt_bill = 0
for b, p in zip(bill_cols, pay_amt_cols):
    pay_gt_bill += (synthetic[p] > synthetic[b]).sum()
print("\n[6] PAY_AMT > BILL_AMT violations:", pay_gt_bill)

bill_gt_limit = 0
for b in bill_cols:
    bill_gt_limit += (synthetic[b] > synthetic["LIMIT_BAL"]).sum()
print("[6] BILL_AMT > LIMIT_BAL violations:", bill_gt_limit)

# =============================
# 7. DEFAULT LOGIC CHECK
# =============================
high_delay_no_default = synthetic[
    (synthetic[pay_cols].max(axis=1) >= 2) &
    (synthetic["default.payment.next.month"] == 0)
]
print("\n[7] High delay but no default:", len(high_delay_no_default))

# =============================
# 8. DISTRIBUTION CHECKS
# =============================
print("\n[8] Default distribution:")
print(synthetic["default.payment.next.month"].value_counts(normalize=True))

print("\n[8] SEX distribution:")
print(synthetic["SEX"].value_counts(normalize=True))

print("\n[8] EDUCATION distribution:")
print(synthetic["EDUCATION"].value_counts(normalize=True))

print("\n[8] MARRIAGE distribution:")
print(synthetic["MARRIAGE"].value_counts(normalize=True))

print("\n[8] PAY_0 distribution:")
print(synthetic["PAY_0"].value_counts(normalize=True).sort_index())

print("\n[8] AGE mean:", synthetic["AGE"].mean())
print("[8] LIMIT_BAL mean:", synthetic["LIMIT_BAL"].mean())
print("[8] LIMIT_BAL median:", synthetic["LIMIT_BAL"].median())

# =============================
# 9. SUMMARY
# =============================
print("\n===================================")
print("SUMMARY")
print("===================================")
print(f"Exact matches with real data: {len(merged)}")
print(f"Duplicate rows: {duplicates}")
print(f"Missing values: {missing}")
print(f"Rows with '?': {question_marks}")
print(f"Invalid AGE rows: {invalid_age}")
print(f"Invalid LIMIT_BAL rows: {invalid_limit}")
print(f"Invalid PAY_X values: {invalid_pay}")
print(f"PAY > BILL violations: {pay_gt_bill}")
print(f"BILL > LIMIT violations: {bill_gt_limit}")
print(f"High delay no default: {len(high_delay_no_default)}")

print("\nDONE.")