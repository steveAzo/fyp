# When Does Detector Choice Matter? An Information-Based Perspective on Multivariate Drift Detection Under Resource Constraints

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete replication code, experimental configuration, and processed run-level datasets for the study:

> **"When Does Detector Choice Matter? An Information-Based Perspective on Multivariate Drift Detection Under Resource Constraints"**  
> *Under review at Statistical Analysis and Data Mining: An ASA Data Science Journal.*  
> **Authors:** Stephen Kwaku Azongo, Kofi Sarpong Adu-Manu, Livingston Seble, Alex Teye Ametepey, Sandro Amofa  
> **Affiliation:** Department of Computer Science, University of Ghana  

---

## 📌 Executive Summary

Practitioners deploying data drift monitoring in production machine learning systems face a fundamental choice: **which detector to deploy**, and **under what resource constraints** (detection window size, monitoring inspection interval, reference window size).

Prior benchmark studies (e.g., Rabanser et al., NeurIPS 2019) observed that operational constraints dominate detector choice when shifts alter individual feature marginals. However, this raises an open question: **does detector choice remain subordinate when distribution shift is strictly confined to cross-feature dependence, leaving all univariate marginals unchanged?**

This benchmark investigates this question across **19,440 experimental runs** (16,200 marginal-shift runs + 3,240 dependency-shift runs) across three tabular domains (Credit Card Default, Telco Customer Churn, and AI4I 2020 Predictive Maintenance):

1. **Information-Based Taxonomy**: Conventional labels ("univariate" vs. "multivariate") fail to predict detection sensitivity. Formally multivariate methods that aggregate per-feature 1D metrics (such as HDDDM, based on per-feature Hellinger distances) suffer identical failure rates (53.3% miss rate) to univariate baselines (KS 57.6%, PSI 57.6%, Wasserstein 60.6%) when marginals are preserved. We establish a distinction between **marginal-information** detectors and **joint-dependence** detectors (e.g., MMD).
2. **The Conditional Inversion**: Under marginal-visible drift, monitoring interval and window size multiply detection delay by ~5×, dominating algorithm choice. Under dependency-only drift, detector construction becomes decisive: MMD reduces the odds of missing drift by 96.8% relative to Kolmogorov-Smirnov (adjusted OR = 0.032, $p < 0.001$).
3. **Resolving Survivorship Bias**: In naive detected-only comparisons, MMD appears slower than marginal detectors because marginal detectors miss >53% of cases and only detect the fastest random false-positive anomalies. Right-censored survival analysis (Kaplan-Meier, log-rank $\chi^2 = 394.69, p < 10^{-80}$) proves that MMD detects drift significantly faster and more completely than all competitors.
4. **Predictive Impact & Class Imbalance**: Counterfactual-matched downstream degradation shows that statistical detectability does not directly imply predictive harm. Evaluating balanced accuracy and ROC-AUC uncovers minority-class vulnerability in imbalanced domains (AI4I Maintenance, 3.4% positive) that plain classification accuracy masks.

---

## 📁 Repository Architecture

```text
├── config.yaml                      # Global experiment configurations & dataset definitions
├── requirements.txt                 # Python dependencies
├── smoke_test.py                    # Pipeline validation script
├── run_experiments.py               # Main experiment execution runner
├── run_mixed_model.py               # Clustered GEE & Cox survival regression (Tables 4 & 5)
├── run_survival_analysis.py         # Kaplan-Meier survival curves & log-rank tests (Table 6, Fig 7)
├── run_predictive_impact.py         # Counterfactual degradation with 95% CIs (Table 7, Fig 6)
├── run_supplementary.py             # False alarm rates, MMD budget, C2ST sensitivity, saturation
├── run_split_sensitivity.py         # Reference/stream split sensitivity analysis (50/50, 70/30)
├── run_feature_pair_sensitivity.py  # Alternative feature-pair sensitivity validation
├── generate_plots.py                # Main manuscript visualization generator
│
├── data/                            # Dataset loaders and preprocessed tabular datasets
│   ├── loader.py                    # Stratified train/test splitting and loading logic
│   ├── credit/                      # UCI Default of Credit Card Clients (N=30,000)
│   ├── telco/                       # Telco Customer Churn (N=7,032)
│   └── maintenance/                 # AI4I 2020 Predictive Maintenance (N=10,000)
│
├── detectors/                       # Drift detector implementations
│   ├── base.py                      # Abstract detector interface & result structures
│   ├── ks_test.py                   # Bonferroni-corrected Kolmogorov-Smirnov test
│   ├── psi.py                       # Population Stability Index (histogram divergence)
│   ├── wasserstein.py               # 1D Wasserstein distance with permutation bootstrap
│   ├── mmd.py                       # Maximum Mean Discrepancy (RBF kernel, permutation test)
│   ├── c2st.py                      # Classifier Two-Sample Test (Logistic Regression & Random Forest)
│   └── hdddm.py                     # Hellinger Distance-Based Drift Detection Method (Ditzler & Polikar)
│
├── simulators/                      # Drift simulation modules
│   ├── base.py                      # Abstract base simulator
│   ├── sudden.py                    # Abrupt step shift
│   ├── gradual.py                   # Sigmoidal linear ramp shift
│   ├── recurring.py                 # Periodic cyclic shift
│   ├── multivariate_drift.py        # Independent joint marginal shift
│   └── correlation_drift.py         # Marginal-preserving rank-permutation copula shift
│
├── models/                          # Downstream prediction models
│   └── trainer.py                   # Random Forest, Logistic Regression, LightGBM training wrapper
│
├── experiments/                     # Experiment grid definition and runner
│   ├── config.py                    # ExperimentConfig dataclass
│   ├── grid.py                      # Factorial grid generators (marginal & correlation grids)
│   └── runner.py                    # Safe batch execution engine with crash recovery
│
├── paper/                           # Paper drafts and figures
│   ├── paper_draft_revision_v1.md   # Full manuscript revision addressing editorial review
│   └── figures/                     # High-resolution figures (300 DPI)
│
└── results/                         # Processed run-level results (19,440 total runs)
    ├── results.csv                  # 16,200 marginal-shift runs
    ├── results_correlation.csv      # 3,240 dependency-shift runs
    ├── significance/                # Mixed model, GEE, and survival analysis outputs
    ├── predictive_impact/           # Counterfactual degradation runs and 95% CIs
    └── supplementary/               # Type I error / false alarm rates, sensitivity runs
```

---

## 🛠️ Quick Start & Installation

### 1. Environment Setup

Python 3.10+ is recommended. Clone this repository and install dependencies:

```bash
git clone https://github.com/YourUsername/multivariate-drift-benchmark.git
cd multivariate-drift-benchmark

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # On Linux/macOS
# .\venv\Scripts\Activate.ps1   # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Sanity Smoke Test

Verify that all 6 detectors, 6 drift simulators, and model trainers execute properly:

```bash
python smoke_test.py
```

Expected output: `[ALL PASSED] Pipeline validated successfully.`

---

## 📊 Replicating Paper Tables & Figures

All processed run-level datasets are pre-included in `results/`, allowing you to reproduce every statistical table and figure in the manuscript in seconds:

### 1. Clustered Mixed-Model & GEE Inference (Tables 4 & 5)
Computes GEE logistic regression models (M1, M2, M3b) and log-linear delay models (M4) clustered over 54 blocks, plus clustered Cox proportional hazards survival regression (M5):
```bash
python run_mixed_model.py
```
*Outputs saved to `results/significance/mixed_model/`.*

### 2. Kaplan-Meier Survival Analysis & Log-Rank Tests (Table 6 & Figure 7)
Computes right-censored time-to-detection statistics, restricted mean survival times (RMST), overall and pairwise log-rank tests, and renders Figure 7:
```bash
python run_survival_analysis.py
```
*Figure saved to `paper/figures/fig7_kaplan_meier_survival.png`.*

### 3. Predictive Impact with 95% Confidence Intervals (Table 7 & Figure 6)
Evaluates counterfactual degradation on Plain Accuracy, Balanced Accuracy, and ROC-AUC across imbalanced datasets with exact 95% CIs:
```bash
python run_predictive_impact.py
```
*Figure saved to `paper/figures/fig6_predictive_impact_multimetric.png`.*

### 4. Split Sensitivity & Feature-Pair Sensitivity (Tables B4, B5, B6)
Validates that findings are invariant to 50/50 and 70/30 reference splits and alternative feature pair selections:
```bash
python run_split_sensitivity.py
python run_feature_pair_sensitivity.py
```

### 5. Type I Error Rates & Supplementary Robustness (Tables 3, B1, B2, B3)
Replicates false-alarm rate benchmarks on no-drift streams, MMD sample budget analysis, and C2ST classifier comparisons:
```bash
python run_supplementary.py --only false_alarm
python run_supplementary.py --only mmd_budget
python run_supplementary.py --only c2st_sensitivity
python run_supplementary.py --only dep_strength
python run_supplementary.py --only saturation
```

---

## 🔬 Re-Running Raw Experiment Grids (From Scratch)

If you wish to re-execute the underlying experimental runs:

```bash
# 1. Quick validation slice (36 configurations):
python run_experiments.py --mode quick

# 2. Full marginal-shift grid (16,200 runs across 3 datasets x 6 detectors x 3 drift types x 5 frequencies x 4 windows x 3 reference sizes x 5 reps):
python run_experiments.py --mode full

# 3. Full dependency-shift grid (3,240 runs across 3 datasets x 6 detectors x 3 drift types x 3 frequencies x 2 windows x 2 reference sizes x 5 reps):
python run_experiments.py --mode correlation
```

Execution is fully resumable; if interrupted, restarting skips already-completed runs automatically.

---

## 📑 Benchmark Datasets

All datasets are real-world tabular classification benchmarks widely cited in drift detection literature:

1. **Default of Credit Card Clients** ($N = 30,000$, 23 features, 22.1% default rate):  
   Yeh, I.-C., & Lien, C.-H. (2009). *Expert Systems with Applications*, 36(2), 2473–2480.  
   Drift features: `AGE` + `BILL_AMT1`.
2. **Telco Customer Churn** ($N = 7,032$, 19 features, 26.5% churn rate):  
   IBM Analytics Community (2018).  
   Drift features: `Monthly Charges` + `Total Charges`.
3. **AI4I 2020 Predictive Maintenance** ($N = 10,000$, 6 features, 3.4% failure rate):  
   Matzka, S. (2020). *IEEE AI4I*, pp. 69–74.  
   Drift features: `Tool wear [min]` + `Air temperature [K]`.

---

## 📜 Citation

If you use this benchmark, code, or experimental data in your research, please cite:

```bibtex
@article{azongo2026detectorchoice,
  title   = {When Does Detector Choice Matter? An Information-Based Perspective on Multivariate Drift Detection Under Resource Constraints},
  author  = {Azongo, Stephen Kwaku and Adu-Manu, Kofi Sarpong and Seble, Livingston and Ametepey, Alex Teye and Amofa, Sandro},
  journal = {Statistical Analysis and Data Mining: An ASA Data Science Journal},
  year    = {2026},
  note    = {Under Review},
  doi     = {10.5281/zenodo.placeholder}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
