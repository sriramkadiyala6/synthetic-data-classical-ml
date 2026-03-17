# Does Training on LLM-Generated Synthetic Data Help or Hurt Classical ML?

**CSCI 567 — Machine Learning | Spring 2026 | University of Southern California**

## Team
- Emon Steadman (esteadma@usc.edu)
- Sriram Kadiyala (skadiyal@usc.edu)
- FNU Nisarga Bhaskar (nisargab@usc.edu)
- Vikash Churiwala (vchuriwa@usc.edu)

---

## Overview

The "model collapse" phenomenon where models trained on synthetic data degrade over generations is well-studied for large language models. But what happens when you use LLM-generated synthetic data to train *classical* ML models like logistic regression, SVMs, or XGBoost? This project systematically investigates whether LLM-generated synthetic tabular data can substitute for, augment, or ultimately hurt classical ML performance compared to real data and traditional synthetic data generation methods.

## Research Questions

1. Does training classical ML models on LLM-generated synthetic tabular data help, hurt, or match training on real data?
2. How does the answer change with dataset size, synthetic-to-real ratio, and model type?
3. How does LLM-generated synthetic data compare against purpose-built synthetic data methods (SMOTE, CTGAN)?

## Datasets

We evaluate across three well-known tabular classification datasets from the UCI Machine Learning Repository:

| Dataset | Samples | Features | Task |
|---------|---------|----------|------|
| [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) | ~48,842 | 14 | Income >50K prediction |
| [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 | 23 | Default prediction |
| [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | ~303 | 13 | Disease presence prediction |

## Synthetic Data Generation Methods

### LLM-Based Generation
- **Model:** Claude Sonnet 4.6 (via Anthropic API)
- **Approach:** Structured prompting with schema descriptions, feature distributions, and sample rows to generate realistic synthetic tabular data
- **Validation:** Statistical checks on generated data (range validation, distribution matching, inter-column relationship preservation)

### Traditional Methods
- **SMOTE** (Synthetic Minority Oversampling Technique) — interpolation-based oversampling via `imbalanced-learn`

## Models

| Model | Library | Course Topic |
|-------|---------|--------------|
| Logistic Regression | scikit-learn | Linear Models, Regularization |
| SVM (RBF Kernel) | scikit-learn | Kernel Methods, SVM |
| XGBoost | xgboost | Decision Trees, Boosting, Ensembles |
| MLP (Multi-Layer Perceptron) | scikit-learn / PyTorch | Neural Networks |

## Experimental Design

### Notation
- **RD** = Real Data
- **SD** = Synthetic Data (LLM-generated, SMOTE, or CTGAN/TVAE)

### Experiment 1: Full Data Regime (Substitution)
All real training data is available. We test whether replacing portions with synthetic data changes performance.

| Condition | RD : SD Ratio |
|-----------|---------------|
| Baseline (real only) | 1.0 / 0.0 |
| Mostly real | 0.75 / 0.25 |
| Equal mix | 0.50 / 0.50 |
| Mostly synthetic | 0.25 / 0.75 |
| Synthetic only | 0.0 / 1.0 |

### Experiment 2: Low Data Regime (Augmentation)
Only a small fraction of real data is available (max of 10% of original samples or number of dimensions). We test whether adding synthetic data helps compensate.

| Condition | Description |
|-----------|-------------|
| Baseline | Limited real data only |
| Augmented | Limited real data + synthetic data at varying amounts |

Same RD:SD ratios as Experiment 1, applied to the reduced real data pool.

### Experiment 3: Comparison Across Synthetic Methods
For each experiment above, repeat using:
- LLM-generated synthetic data
- SMOTE synthetic data
- CTGAN synthetic data
- TVAE synthetic data (extension)

### Hyperparameter Tuning
- Independent cross-validation for each RD:SD ratio and synthetic method
- Prevents unfair advantage from hyperparameters tuned on a different data condition

### Evaluation
- All models evaluated on the **same held-out real test set** (fixed 80/20 split at the start)
- **Metrics:** Accuracy, F1 Score, AUC-ROC

## Project Structure

```
synthetic-data-classical-ml/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Original datasets
│   ├── synthetic/
│   │   ├── llm/                # LLM-generated synthetic data
│   │   ├── smote/              # SMOTE-generated synthetic data
│   │   └── ctgan/              # CTGAN-generated synthetic data
│   └── processed/              # Train/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_llm_generation.ipynb
│   ├── 03_smote_generation.ipynb
│   ├── 04_ctgan_generation.ipynb
│   ├── 05_experiments.ipynb
│   └── 06_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── llm_generator.py
│   ├── smote_generator.py
│   ├── ctgan_generator.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── utils.py
├── results/
│   ├── tables/
│   └── figures/
└── .gitignore
```

## Setup

```bash
git clone https://github.com/<your-username>/synthetic-data-classical-ml.git
cd synthetic-data-classical-ml
pip install -r requirements.txt
```

### Dependencies
- Python 3.10+
- scikit-learn
- xgboost
- imbalanced-learn (for SMOTE)
- anthropic (for Claude API)
- pandas, numpy, matplotlib, seaborn

## Timeline

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| Proposal submitted | March 13, 2026 | Project proposal |
| TA check-in presentation | Week of March 23-27, 2026 | 5-slide presentation |
| Pre-final check-in | Week of April 20-24, 2026 | Simulations complete, preliminary results |
| Final report | Finals week (TBD) | 5-6 page report with analysis |

## References

1. Borisov, V., et al. "Language Models are Realistic Tabular Data Generators." ICLR, 2023. (GReaT)
2. Xu, L., et al. "Modeling Tabular data using Conditional GAN." NeurIPS, 2019. (CTGAN)
3. Chawla, N.V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." JAIR, 2002.
4. Fang, Y., et al. "Large Language Models on Tabular Data — A Survey." arXiv:2402.17944, 2024.
5. Singh, A., et al. "Are LLMs Naturally Good at Synthetic Tabular Data Generation?" arXiv:2406.14541, 2024.
6. Wen, Z., et al. "HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection." NeurIPS D&B, 2024.

## License

This project is for academic purposes as part of CSCI 567 at USC.
