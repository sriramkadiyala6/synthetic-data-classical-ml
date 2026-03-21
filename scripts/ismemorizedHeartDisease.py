import pandas as pd

df = pd.read_csv("heart_disease_synthetic_combined.csv")
real = pd.read_csv("processed.cleveland.data", header=None, na_values='?',
                    names=['age','sex','cp','trestbps','chol','fbs','restecg',
                           'thalach','exang','oldpeak','slope','ca','thal','target'])
# Binarize real target
real['target'] = (real['target'] > 0).astype(int)

print("===================================")
print("DATA VALIDATION REPORT")
print("===================================")

# 0. Memorization
merged = df.merge(real.drop_duplicates(), how='inner')
print("\n[0] Exact matches with real data:", len(merged))

# 1. Duplicates
print("[1] Duplicates:", df.duplicated().sum())

# 2. Missing
print("[2] Missing:", df.isnull().sum().sum())

# 3. Ranges
print("\n[3] Invalid age:", ((df["age"] < 29) | (df["age"] > 77)).sum())
print("[3] Invalid trestbps:", ((df["trestbps"] < 90) | (df["trestbps"] > 200)).sum())
print("[3] Invalid chol:", ((df["chol"] < 120) | (df["chol"] > 600)).sum())
print("[3] Invalid thalach:", ((df["thalach"] < 70) | (df["thalach"] > 210)).sum())
print("[3] Invalid oldpeak:", ((df["oldpeak"] < 0) | (df["oldpeak"] > 6.2)).sum())

# 4. Categories
print("\n[4] Invalid sex:", (~df["sex"].isin([0,1])).sum())
print("[4] Invalid cp:", (~df["cp"].isin([1,2,3,4])).sum())
print("[4] Invalid fbs:", (~df["fbs"].isin([0,1])).sum())
print("[4] Invalid restecg:", (~df["restecg"].isin([0,1,2])).sum())
print("[4] Invalid exang:", (~df["exang"].isin([0,1])).sum())
print("[4] Invalid slope:", (~df["slope"].isin([1,2,3])).sum())
print("[4] Invalid ca:", (~df["ca"].isin([0,1,2,3])).sum())
print("[4] Invalid thal:", (~df["thal"].isin([3,6,7])).sum())
print("[4] Invalid target:", (~df["target"].isin([0,1])).sum())

# 5. Distributions
print("\n[5] Target distribution:")
print(df["target"].value_counts(normalize=True))
print("\nSex distribution:")
print(df["sex"].value_counts(normalize=True))
print("\ncp distribution:")
print(df["cp"].value_counts(normalize=True).sort_index())
print("\nthal distribution:")
print(df["thal"].value_counts(normalize=True).sort_index())
print("\nca distribution:")
print(df["ca"].value_counts(normalize=True).sort_index())
print("\nAge mean:", df["age"].mean())
print("Chol mean:", df["chol"].mean())
print("Thalach mean:", df["thalach"].mean())
print("Oldpeak mean:", df["oldpeak"].mean())
print("Trestbps mean:", df["trestbps"].mean())

# 6. Summary
print("\n===================================")
print("SUMMARY")
print("===================================")
print(f"Exact matches: {len(merged)}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Missing: {df.isnull().sum().sum()}")
print("\nDONE.")