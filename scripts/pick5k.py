#pick 5000 rows for the adult dataset

import pandas as pd

df = pd.read_csv("adult_synthetic_combined.csv")
sample = df.sample(n=5000, random_state=42)
sample.to_csv("adult_synthetic_5000.csv", index=False)
print(sample.shape)