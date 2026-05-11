from lib.preprocess import *
from sklearn.model_selection import train_test_split
import os

NUM_ROWS = 5000
RANDOM_SEED = 42
TEST_SIZE = 0.2

'''
Summary of data:
Training n rows:    X_tr_trim, y_tr_trim
Test data:          X_ts, y_ts

Synthetic n rows:   X_syn, y_syn
SMOTE n rows:       X_smote, y_smote
CTGAN n rows:       X_ctgan, y_ctgan
'''

################################################################
# Adult income data set
################################################################

# Import raw train and tset data
X_tr_raw, y_tr_raw = import_adult('data/raw/adult/adult.data')
X_ts_raw, y_ts = import_adult('data/raw/adult/adult.test', num_skip_rows=1)

# Import synthetic & ctgan data
X_syn_raw, y_syn = import_adult('data/synthetic/LLM/Adult Dataset Files/adult_synthetic_5000.csv', num_skip_rows=1)
X_ctgan_raw, y_ctgan = load_ctgan('data/synthetic/CTGAN/adult_ctgan.csv')

# Create and apply encoder to process data to ML format
X_tr, encoder = preprocess_adult(X_tr_raw)
X_ts = preprocess_adult_encoder(X_ts_raw, encoder)
X_syn = preprocess_adult_encoder(X_syn_raw, encoder)
X_ctgan = preprocess_adult_encoder(X_ctgan_raw, encoder)

# Trim down raw data (Random sample of rows to avoid ordering biases)
X_tr_trim, y_tr_trim = sample_n_rows(X_tr, y_tr_raw, n=NUM_ROWS)

# Generate SMOTE (runs on processed data)
## Consider removing because it only generates classes of imbalance, otherwise, will need to artificially create imbalanced set to work on, and re-combine.
#X_smote, y_smote =  get_smote_data(X_tr, y_tr_raw, n=NUM_ROWS)


################################################################
# Credit card default data
################################################################

# Import raw data, create Train/Test split
X_raw, y_raw = import_credit('data/raw/credit/default of credit card clients.csv')
X_tr_raw, X_ts_raw, y_tr_raw, y_ts_ = train_test_split(X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_raw)

# Import synthetic & ctgan data
X_syn_raw, y_syn = import_credit_synthetic('data/synthetic/LLM/Default of Credit Card Holders Dataset Files/credit_synthetic_combined.csv')
X_ctgan_raw, y_ctgan = load_ctgan('data/synthetic/CTGAN/credit_ctgan.csv')

# Create and apply encoder to process data to ML format
X_tr, encoder = preprocess_credit(X_tr_raw)
X_ts = preprocess_credit_encoder(X_ts_raw, encoder)
X_syn = preprocess_credit_encoder(X_syn_raw, encoder)
X_ctgan = preprocess_credit_encoder(X_ctgan_raw, encoder)

# Trim the raw data down to 5000 samples to simulate data scarcity
X_tr_trim, y_tr_trim = sample_n_rows(X_tr, y_tr_raw, n=NUM_ROWS)

# Generate SMOTE (runs on processed data)
## Consider removing because it only generates classes of imbalance, otherwise, will need to artificially create imbalanced set to work on, and re-combine.
#X_smote, y_smote =  get_smote_data(X_tr, y_tr_raw, n=NUM_ROWS)

################################################################
# Heart data
################################################################

# Import raw data, create Train/Test split
base_path = 'data/raw/heart+disease'
paths = [
    os.path.join(base_path, 'processed.cleveland.data'),
    os.path.join(base_path, 'processed.hungarian.data'),
    os.path.join(base_path, 'processed.switzerland.data'),
    os.path.join(base_path, 'processed.va.data')
]
X_raw, y_raw = import_and_combine_heart(paths, impute=True)
X_tr_raw, X_ts_raw, y_tr_raw, y_ts = train_test_split(X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_raw)

heart_n = len(X_tr_raw)

# Import synthetic & ctgan data
X_syn_raw, y_syn = import_heart_synthetic('data/synthetic/LLM/Heart Disease Dataset Files/heart_disease_synthetic_combined.csv')
X_ctgan_raw, y_ctgan = load_ctgan('data/synthetic/CTGAN/heart_ctgan.csv')

# Create and apply encoder to process data to ML format
X_tr, encoder = preprocess_heart(X_tr_raw)
X_ts = preprocess_heart_encoder(X_ts_raw, encoder)
X_syn_enc = preprocess_heart_encoder(X_syn_raw, encoder)
X_syn, y_syn = sample_n_rows(X_syn_enc, y_syn, n=heart_n)
X_ctgan_enc = preprocess_heart_encoder(X_ctgan_raw, encoder)
X_ctgan, y_ctgan = sample_n_rows(X_syn_enc, y_ctgan, n=heart_n)

# Trim down training data
X_tr_trim, y_tr_trim = sample_n_rows(X_tr, y_tr_raw, n=heart_n)

# Generate SMOTE (runs on processed data)
## Consider removing because it only generates classes of imbalance, otherwise, will need to artificially create imbalanced set to work on, and re-combine.
#X_smote, y_smote = get_smote_data(X_tr, y_tr_raw, n=NUM_ROWS)
