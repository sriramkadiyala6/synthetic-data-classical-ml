import pandas as pd
import glob

files = sorted(glob.glob("adult_incom*.csv"))

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df.columns) == 15:
            dfs.append(df)
        else:
            print(f"Skipping {f} — wrong column count: {len(df.columns)}")
    except:
        try:
            df = pd.read_csv(f, header=None)
            df.columns = ['age','workclass','fnlwgt','education','education-num',
                          'marital-status','occupation','relationship','race','sex',
                          'capital-gain','capital-loss','hours-per-week','native-country','income']
            if len(df.columns) == 15:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates()
print(f"Total rows: {len(combined)}")
print(f"Columns: {len(combined.columns)}")
print(combined.head())
combined.to_csv("adult_synthetic_combined.csv", index=False)
print("Saved.")