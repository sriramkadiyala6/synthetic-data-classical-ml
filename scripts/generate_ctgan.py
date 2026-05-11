import sys
sys.path.append('../')
from lib.preprocess import *
from sklearn.model_selection import train_test_split
import os

NUM_ROWS = 5000
RANDOM_SEED = 42
TEST_SIZE = 0.2

def save_synthetic(X, y, path):
    df = X.copy()
    df['__target__'] = y.values
    df.to_csv(path, index=False)

os.makedirs('../data/synthetic/CTGAN', exist_ok=True)

################################################################
# Adult income
################################################################
print("Generating CTGAN data for Adult dataset...")
X_tr_raw, y_tr_raw = import_adult('../data/raw/adult/adult.data')

X_ctgan_raw, y_ctgan = get_ctgan_data(X_tr_raw, y_tr_raw, n=NUM_ROWS)
save_synthetic(X_ctgan_raw, y_ctgan, '../data/synthetic/CTGAN/adult_ctgan.csv')
print("Saved adult_ctgan.csv")

################################################################
# Credit card default
################################################################
print("Generating CTGAN data for Credit dataset...")
X_raw, y_raw = import_credit('../data/raw/default/default of credit card clients.csv')
X_tr_raw, _, y_tr_raw, _ = train_test_split(X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_raw)

X_ctgan_raw, y_ctgan = get_ctgan_data(X_tr_raw, y_tr_raw, n=NUM_ROWS, discrete_columns=["SEX", "EDUCATION", "MARRIAGE"])
save_synthetic(X_ctgan_raw, y_ctgan, '../data/synthetic/CTGAN/credit_ctgan.csv')
print("Saved credit_ctgan.csv")

################################################################
# Heart data
################################################################
print("Generating CTGAN data for Heart dataset...")
base_path = '../data/raw/heart+disease'
paths = [
    os.path.join(base_path, 'processed.cleveland.data'),
    os.path.join(base_path, 'processed.hungarian.data'),
    os.path.join(base_path, 'processed.switzerland.data'),
    os.path.join(base_path, 'processed.va.data')
]

X_raw, y_raw = import_and_combine_heart(paths)
X_tr_raw, _, y_tr_raw, _ = train_test_split(X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_raw)

X_ctgan_raw, y_ctgan = get_ctgan_data(X_tr_raw, y_tr_raw, n=NUM_ROWS, discrete_columns=['cp', 'restecg', 'slope', 'thal'])
save_synthetic(X_ctgan_raw, y_ctgan, '../data/synthetic/CTGAN/heart_ctgan.csv')
print("Saved heart_ctgan.csv")

print("Done! All CTGAN data saved to ../data/synthetic/CTGAN/")
