import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sys
from sklearn.neural_network import MLPClassifier

sys.path.append("../lib")
from preprocess import (
    import_adult, preprocess_adult, preprocess_adult_encoder,
    import_credit, import_credit_synthetic, preprocess_credit, preprocess_credit_encoder,
    import_and_combine_heart, import_heart_synthetic, preprocess_heart, preprocess_heart_encoder,
    sample_n_rows, combine_datasets, load_ctgan
)

# ============ PATHS ============
# Raw UCI data
ADULT_TRAIN_PATH = "../data/raw/adult/adult.data"
ADULT_TEST_PATH = "../data/raw/adult/adult.test"
CREDIT_PATH = "../data/raw/credit/default of credit card clients.csv"
HEART_PATH = "../data/raw/heart+disease/processed.cleveland.data"

# LLM synthetic — distribution constrained
ADULT_LLM_CONSTRAINED = "../data/Synthetic/LLM/Adult Dataset Files/adult_synthetic_5000.csv"
CREDIT_LLM_CONSTRAINED = "../data/Synthetic/LLM/Default of Credit Card Holders Dataset Files/credit_synthetic_combined.csv"
HEART_LLM_CONSTRAINED = "../data/Synthetic/LLM/Heart Disease Dataset Files/heart_disease_synthetic_combined.csv"

# LLM synthetic — sample rows, no constraints
ADULT_LLM_SAMPLE = "../data/Synthetic/LLM/Adult Dataset Files/adult_sample_combined.csv"
CREDIT_LLM_SAMPLE = "../data/Synthetic/LLM/Default of Credit Card Holders Dataset Files/credit_sample_combined.csv"
HEART_LLM_SAMPLE = "../data/Synthetic/LLM/Heart Disease Dataset Files/heart_sample_combined_clean.csv"

# CTGAN
ADULT_CTGAN = "../data/Synthetic/CTGAN/adult_ctgan.csv"
CREDIT_CTGAN = "../data/Synthetic/CTGAN/credit_ctgan.csv"
HEART_CTGAN = "../data/Synthetic/CTGAN/heart_ctgan.csv"

# ============ CONFIG ============
SEEDS = [42, 123, 456, 789, 1001]
RATIOS = [1.0, 0.75, 0.5, 0.25, 0.0]
FULL_TRAIN_SIZE = 5000
LOW_DATA_FRACTION = 0.10

results = []


def run_experiments(dataset_name, X_real_train, y_real_train, X_test, y_test,
                    synthetic_sources, train_size, regime):
    """Run all ratio x method x seed experiments for a single dataset/regime."""
    for method_name, X_synth, y_synth in synthetic_sources:
        for ratio in RATIOS:
            for seed in SEEDS:
                try:
                    # Handle edge cases
                    if ratio == 1.0:
                        X_train, y_train = sample_n_rows(X_real_train, y_real_train, train_size, seed=seed)
                    elif ratio == 0.0:
                        X_train, y_train = sample_n_rows(X_synth, y_synth, train_size, seed=seed)
                    else:
                        X_train, y_train = combine_datasets(
                            X_real_train, y_real_train, X_synth, y_synth,
                            pct_from_1=ratio, seed=seed
                        )
                        X_train, y_train = sample_n_rows(X_train, y_train, train_size, seed=seed)

                    # Align columns (synthetic may be missing some after OHE)
                    X_train = X_train.reindex(columns=X_test.columns, fill_value=0)
                    # Fill any NaN values (CTGAN sometimes produces them)
                    X_train = X_train.fillna(X_train.median(numeric_only=True))

                    # Scale
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train
                    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-4, batch_size='auto', learning_rate_init=1e-3, max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=seed)

                    model.fit(X_train_scaled, y_train)

                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]

                    results.append({
                        "dataset": dataset_name,
                        "regime": regime,
                        "model": "mlp",
                        "method": method_name,
                        "ratio": ratio,
                        "seed": seed,
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1": f1_score(y_test, y_pred),
                        "auc": roc_auc_score(y_test, y_proba),
                    })
                    print(f"{dataset_name} {regime} {method_name} ratio={ratio} seed={seed} done")
                except Exception as e:
                    print(f"FAIL: {dataset_name} {regime} {method_name} ratio={ratio} seed={seed}: {e}")


# ============ ADULT ============
print("=" * 60)
print("ADULT")
print("=" * 60)

X_real, y_real = import_adult(ADULT_TRAIN_PATH)
X_test_raw, y_test = import_adult(ADULT_TEST_PATH, num_skip_rows=1)

X_real_enc, encoder = preprocess_adult(X_real)
X_test_enc = preprocess_adult_encoder(X_test_raw, encoder)

X_llm_c, y_llm_c = import_adult(ADULT_LLM_CONSTRAINED, num_skip_rows=1)
X_llm_c_enc = preprocess_adult_encoder(X_llm_c, encoder)

X_llm_s, y_llm_s = import_adult(ADULT_LLM_SAMPLE, num_skip_rows=1)
X_llm_s_enc = preprocess_adult_encoder(X_llm_s, encoder)

X_ctgan_raw, y_ctgan = load_ctgan(ADULT_CTGAN)
X_ctgan = preprocess_adult_encoder(X_ctgan_raw, encoder)

synthetic_sources = [
    ("llm_constrained", X_llm_c_enc, y_llm_c),
    ("llm_sample", X_llm_s_enc, y_llm_s),
    ("ctgan", X_ctgan, y_ctgan),
]

# Full data regime
run_experiments("adult", X_real_enc, y_real, X_test_enc, y_test,
                synthetic_sources, FULL_TRAIN_SIZE, "full")

# Low data regime
low_n = min(int(len(X_real_enc) * LOW_DATA_FRACTION), 500)

X_real_low, y_real_low = sample_n_rows(X_real_enc, y_real, low_n, seed=42)
run_experiments("adult", X_real_low, y_real_low, X_test_enc, y_test,
                synthetic_sources, low_n * 2, "low")


# ============ CREDIT ============
print("=" * 60)
print("CREDIT")
print("=" * 60)

X_credit, y_credit = import_credit(CREDIT_PATH)
X_real, X_test_raw, y_real, y_test = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
)

X_real_enc, encoder = preprocess_credit(X_real)
X_test_enc = preprocess_credit_encoder(X_test_raw, encoder)

X_llm_c, y_llm_c = import_credit_synthetic(CREDIT_LLM_CONSTRAINED)
X_llm_c_enc = preprocess_credit_encoder(X_llm_c, encoder)

X_llm_s, y_llm_s = import_credit_synthetic(CREDIT_LLM_SAMPLE)
X_llm_s_enc = preprocess_credit_encoder(X_llm_s, encoder)

X_ctgan_raw, y_ctgan = load_ctgan(CREDIT_CTGAN)
X_ctgan = preprocess_credit_encoder(X_ctgan_raw, encoder)

synthetic_sources = [
    ("llm_constrained", X_llm_c_enc, y_llm_c),
    ("llm_sample", X_llm_s_enc, y_llm_s),
    ("ctgan", X_ctgan, y_ctgan),
]

run_experiments("credit", X_real_enc, y_real, X_test_enc, y_test,
                synthetic_sources, FULL_TRAIN_SIZE, "full")

low_n = min(int(len(X_real_enc) * LOW_DATA_FRACTION), 500)
X_real_low, y_real_low = sample_n_rows(X_real_enc, y_real, low_n, seed=42)
run_experiments("credit", X_real_low, y_real_low, X_test_enc, y_test,
                synthetic_sources, low_n * 2, "low")


# ============ HEART ============
print("=" * 60)
print("HEART")
print("=" * 60)

X_heart, y_heart = import_and_combine_heart([HEART_PATH], impute=True)
X_real, X_test_raw, y_real, y_test = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)

X_real_enc, encoder = preprocess_heart(X_real)
X_test_enc = preprocess_heart_encoder(X_test_raw, encoder)

X_llm_c, y_llm_c = import_heart_synthetic(HEART_LLM_CONSTRAINED)
X_llm_c_enc = preprocess_heart_encoder(X_llm_c, encoder)

X_llm_s, y_llm_s = import_heart_synthetic(HEART_LLM_SAMPLE)
X_llm_s_enc = preprocess_heart_encoder(X_llm_s, encoder)

X_ctgan_raw, y_ctgan = load_ctgan(HEART_CTGAN)
X_ctgan = preprocess_heart_encoder(X_ctgan_raw, encoder)

synthetic_sources = [
    ("llm_constrained", X_llm_c_enc, y_llm_c),
    ("llm_sample", X_llm_s_enc, y_llm_s),
    ("ctgan", X_ctgan, y_ctgan),
]

# Heart is small — scale train size to available real data
heart_train_size = min(FULL_TRAIN_SIZE, len(X_real_enc) * 5)
run_experiments("heart", X_real_enc, y_real, X_test_enc, y_test,
                synthetic_sources, heart_train_size, "full")

low_n = min(max(int(len(X_real_enc) * LOW_DATA_FRACTION), 30), 500)
X_real_low, y_real_low = sample_n_rows(X_real_enc, y_real, low_n, seed=42)
run_experiments("heart", X_real_low, y_real_low, X_test_enc, y_test,
                synthetic_sources, low_n * 2, "low")


# ============ SAVE ============
results_df = pd.DataFrame(results)
results_df.to_csv("../results/results_mlp.csv", index=False)
print(f"\nDone. {len(results_df)} results saved.")