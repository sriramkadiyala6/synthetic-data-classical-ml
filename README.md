# Does LLM-Generated Synthetic Data Help or Hurt Classical ML Training?

**CSCI 567 -- Machine Learning | Spring 2026 | University of Southern California**

## Team
- Sriram Kadiyala (skadiyal@usc.edu)
- Emon Steadman (esteadma@usc.edu)
- FNU Nisarga Bhaskar (nisargab@usc.edu)
- Vikash Churiwala (vchuriwa@usc.edu)

---

**Note:** The repository was reorganized for readability after all experiments were completed. As a result, relative file paths referenced in the scripts under `lib/` may not resolve correctly if re-run from the current directory structure. All experimental results (CSVs and plots) are included in the repo and were generated prior to reorganization. Refer to the git history for the original working directory layout.

---

## Overview

We systematically investigate whether LLM-generated synthetic tabular data can substitute for, augment, or hurt classical ML performance compared to real data and CTGAN as a traditional baseline. We compare two LLM prompting strategies (distribution-constrained and sample-based) across five real-to-synthetic mixing ratios in both full-data and low-data regimes.

## Datasets

| Dataset | Samples | Features | Task |
|---------|---------|----------|------|
| [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) | 48,842 | 14 | Income >50K prediction |
| [Credit Card Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 | 23 | Default prediction |
| [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | 13 | Disease presence prediction |

## Models

| Model | Implementation |
|-------|---------------|
| Logistic Regression | scikit-learn LogisticRegression (lbfgs, max_iter=1000) |
| SVM | scikit-learn LinearSVC (max_iter=10000) |
| XGBoost | xgboost (5-fold CV for hyperparameter selection) |
| MLP | Two hidden layers (128, 64), ReLU, Adam, early stopping |

## Synthetic Data Generation

| Method | Description |
|--------|-------------|
| LLM-Constrained | Claude Opus 4.6 prompted with schema + explicit distribution targets |
| LLM-Sample | Claude Opus 4.6 prompted with schema + 100 real sample rows, no distribution constraints |
| CTGAN | Conditional Tabular GAN trained on real data |

## Repository Structure

```
synthetic-data-classical-ml/
|-- README.md
|-- requirements.txt
|
|-- lib/                                    # Model training scripts
|   |-- preprocess.py                       # Data loading, encoding, splitting, mixing
|   |-- logistic_regression.py              # LR experiments (450 runs)
|   |-- svm.py                              # SVM experiments (450 runs)
|   |-- xgb.py                              # XGBoost experiments (450 runs)
|   |-- mlp.py                              # MLP experiments (450 runs)
|   |-- template.py                         # Base template for model scripts
|
|-- scripts/                                # Utility and validation scripts
|   |-- analysis.py                         # Results analysis and plotting
|   |-- generate_ctgan.py                   # CTGAN data generation
|   |-- ismemorizedadult.py                 # Adult synthetic data validation
|   |-- ismemorizedDefaultofcredcard.py     # Credit synthetic data validation
|   |-- ismemorizedHeartDisease.py          # Heart synthetic data validation
|   |-- merge.py                            # Batch merging script
|   |-- pick5k.py                           # Subsample to 5000 rows
|
|-- results/                                # Experimental results
|   |-- results_logistic_regression.csv     # 450 LR runs
|   |-- results_svm.csv                     # 450 SVM runs
|   |-- results_mlp.csv                     # 450 MLP runs
|   |-- xgboost_results.csv                 # 450 XGBoost runs
|
|-- plots/                                  # Result visualizations
|   |-- logistic_regression/                # 6 plots (3 datasets x 2 regimes)
|   |-- svm/                                # 6 plots
|   |-- xgboost/                            # 6 plots
|   |-- mlp/                                # 6 plots
|
|-- data/
|   |-- raw/                                # Original UCI datasets
|   |   |-- adult/
|   |   |   |-- adult.data
|   |   |   |-- adult.test
|   |   |-- credit/
|   |   |   |-- default of credit card clients.xls
|   |   |-- heart+disease/
|   |       |-- processed.cleveland.data
|   |
|   |-- Synthetic/
|   |   |-- CTGAN/
|   |   |   |-- adult_ctgan.csv
|   |   |   |-- credit_ctgan.csv
|   |   |   |-- heart_ctgan.csv
|   |   |
|   |   |-- LLM/
|   |       |-- Adult Dataset Files/
|   |       |   |-- adult_synthetic_5000.csv         # Distribution-constrained (5K rows)
|   |       |   |-- adult_sample_combined.csv        # Sample-based (5K rows)
|   |       |   |-- adult_synthetic_combined.csv     # Full 23K constrained rows
|   |       |   |-- adult_sample_100.csv             # 100 sample rows used in prompt
|   |       |   |-- adult_sample_batch_results.txt   # Batch validation log
|   |       |   |-- Batches/                         # Individual generation batches
|   |       |   |-- Original Dataset/                # Copy of UCI source files
|   |       |
|   |       |-- Default of Credit Card Holders Dataset Files/
|   |       |   |-- credit_synthetic_combined.csv    # Distribution-constrained
|   |       |   |-- credit_sample_combined.csv       # Sample-based
|   |       |   |-- credit_sample_100.csv
|   |       |   |-- credit_sample_batch_results.txt
|   |       |   |-- Batches/
|   |       |
|   |       |-- Heart Disease Dataset Files/
|   |           |-- heart_disease_synthetic_combined.csv  # Distribution-constrained
|   |           |-- heart_sample_combined_clean.csv       # Sample-based (4,125 rows after dedup)
|   |           |-- heart_sample_100.csv
|   |           |-- heart_sample_batch_results.txt
|   |           |-- Batches/
|   |
|   |-- Prompt_log.pdf                      # Documentation of prompts used
|
|-- archive/                                # Original analysis folders (pre-reorganization)
    |-- logreg_plots/
    |-- svm_analysis/
    |-- xgboost/
```

## How to Reproduce Results

**Note:** Due to the post-experiment reorganization, scripts may need path adjustments to run. See git history for original paths.

Each model script in `lib/` runs all 450 experimental conditions (3 datasets x 2 regimes x 3 methods x 5 ratios x 5 seeds):

```bash
cd lib
python logistic_regression.py    # outputs results_logistic_regression.csv
python svm.py                    # outputs results_svm.csv
python xgb.py                    # outputs xgboost_results.csv
python mlp.py                    # outputs results_mlp.csv
```

To generate plots from results:
```bash
python scripts/analysis.py
```

### Dependencies

```bash
pip install -r requirements.txt
```

- Python 3.10+
- scikit-learn
- xgboost
- imbalanced-learn
- ctgan
- pandas, numpy, matplotlib

## Key Findings

- **LLM synthetic data helps in low-data regimes:** On Heart Disease (303 rows), pure LLM synthetic data outperformed real data across all four models (e.g., 0.879 vs 0.787 accuracy for logistic regression).
- **Diminishing returns on large datasets:** On Adult and Credit, synthetic data acts as a neutral substitute with slight degradation.
- **CTGAN collapses on small datasets:** Near-random performance when trained on insufficient real data.
- **Prompting strategy matters:** Distribution constraints improve marginal fidelity; sample-based prompts improve inter-column logical consistency.
- **Model sensitivity varies:** XGBoost is most sensitive to synthetic data quality; logistic regression and SVM are more forgiving.

## References

1. Borisov, V., et al. "Language Models are Realistic Tabular Data Generators." ICLR, 2023.
2. Xu, L., et al. "Modeling Tabular Data using Conditional GAN." NeurIPS, 2019.
3. Seedat, N., et al. "Curated LLM: Synergy of LLMs and Data Curation for Tabular Augmentation in Ultra Low-Data Regimes." NeurIPS, 2024.
4. Grinsztajn, L., et al. "Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?" NeurIPS, 2022.
5. Xu, D., et al. "Are LLMs Naturally Good at Synthetic Tabular Data Generation?" arXiv:2406.14541, 2024.
6. Chen, T. and Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD, 2016.

## License

This project is for academic purposes as part of CSCI 567 at USC.
