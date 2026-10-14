# When Configuration Matters More Than Algorithm

**An Empirical Evaluation of Lightweight Drift Detectors Under Resource Constraints**

Final Year Project, BSc Computer Science, University of Ghana
Stephen Kwaku Azongo

---

## Overview

Machine learning models in production degrade when the data they see drifts away from the data they were trained on. Drift detectors are meant to catch this, but most evaluations of them assume plenty of compute and a fixed monitoring setup. This project asks a practical question instead:

> **When, why, and under what constraints do lightweight drift detection approaches fail or succeed?**

To answer it, three lightweight, distribution-based detectors are run under a full factorial grid of resource constraints on three real-world tabular datasets, with simulated drift injected at a known point so detection delay can be measured exactly.

| Factor | Levels |
|---|---|
| Detectors | Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI), Wasserstein distance |
| Datasets | UCI Credit Card Default, IBM Telco Customer Churn, AI4I 2020 Predictive Maintenance |
| Drift types | Sudden, gradual, recurring |
| Monitoring frequency | Every 10, 50, 100, 500, 1000 samples |
| Window size | 50, 100, 500, 1000 samples |
| Reference size | 100, 500, full reference set |
| Models | Random Forest, Logistic Regression, LightGBM |

The primary experiment has 1,620 configurations × 5 repetitions = **8,100 runs**. A supplementary model comparison experiment adds 405 runs. Everything ran on a single consumer laptop (Intel Core i3, 4 GB RAM), which fits the resource-constrained setting being studied.

## Key Findings

1. **The monitoring setup matters far more than the choice of detector.** Constraint configuration alone produces a 37-fold range in detection delay, much larger than the gap between detectors.
2. **Gradual drift is the hard case.** It is detected significantly later than sudden or recurring drift (r ≈ 0.30, medium effect).
3. **Random Forest is the most robust model under drift.** It degrades significantly less than LightGBM (r = 0.383) or Logistic Regression (r = 0.185), though the gap narrows under imbalance-aware metrics such as AUC.

These combine into a priority order for deployment decisions:

1. **Protect monitoring frequency first.** Avoid checking less often than every 100 samples. At intervals of 1,000, detections are missed structurally.
2. **Tune window size second.** A window of 100 to 500 samples is a sensible operating range.
3. **Keep at least 500 reference samples.** Beyond that, a bigger reference set adds little.
4. **Choose the detector and model last.** Once the monitoring regime is right, all three detectors work. Wasserstein misses the fewest drifts, KS gives the best false positive control, and PSI is the cheapest to compute.

All comparisons use non-parametric tests (Kruskal-Wallis, Mann-Whitney U with Bonferroni correction) with effect sizes (rank-biserial r, Spearman's ρ).

## Repository Structure

```
FYP/
├── 01_literature_review/     Reading notes on drift detection, PSI critique, edge ML constraints
├── 02_implementation/        Experimental framework (Python)
│   ├── data/                 Datasets and loader
│   ├── detectors/            KS, PSI, and Wasserstein detectors behind a common base class
│   ├── simulators/           Sudden, gradual, and recurring drift injection
│   ├── models/               Random Forest, Logistic Regression, LightGBM training
│   ├── experiments/          Experiment config, grid generation, runner, sensitivity studies
│   ├── evaluation/           Metrics and significance testing
│   ├── results/              Raw experiment outputs (CSV)
│   ├── config.yaml           Datasets, thresholds, and constraint grid
│   ├── run_experiments.py    Main entry point
│   ├── smoke_test.py         End-to-end pipeline check
│   └── demo_compare_windows.py  Small demo isolating the effect of window size
├── 03_thesis/                Thesis chapters (Markdown and Word) and paper draft
├── 04_results/
│   ├── analysis/             Scripts that turn raw results into tables and figures
│   ├── experiments/          Summary tables used in the thesis
│   ├── significance/         Statistical test outputs
│   └── plots/                Figures used in the thesis
├── presentation/             Defense slides (LaTeX/Beamer and PowerPoint)
└── proposal.md, outline.md   Original proposal and project plan
```

## Getting Started

### Requirements

Python 3.11 or newer (developed on Python 3.13).

```bash
git clone <this-repo-url>
cd FYP/02_implementation

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

The three datasets are already included in `02_implementation/data/`, so no download step is needed.

### 1. Check the pipeline

Run this first. It loads every dataset, checks every detector and simulator, runs one full experiment, and writes a results row. It takes about 30 seconds.

```bash
python smoke_test.py
```

### 2. Run the experiments

All commands are run from `02_implementation/`.

```bash
# Quick validation grid (about 108 configs, a few minutes)
python run_experiments.py --mode quick

# Full constraint grid (8,100 runs, a few hours on a laptop)
python run_experiments.py --mode full

# Model comparison grid (405 runs)
python run_experiments.py --mode model-comparison

# Restrict to particular datasets
python run_experiments.py --mode full --datasets credit telco
```

Results are appended to `results/results.csv` as runs finish. Re-running resumes where it left off; pass `--no-resume` to start fresh.

### 3. Sensitivity studies (optional)

These check that the model robustness findings are not an artefact of particular hyperparameters or split ratios. Run them from `02_implementation/`:

```bash
python experiments/sensitivity/rf_depth_sensitivity.py
python experiments/sensitivity/lgb_lr_sensitivity.py
python experiments/sensitivity/split_ratio_sensitivity.py
python experiments/sensitivity/model_comparison_extra_metrics.py
```

### 4. Generate tables and figures

```bash
cd ../04_results/analysis
python analyse.py
```

This writes the summary tables to `04_results/experiments/` and the figures to `04_results/plots/`.

### Demo

`demo_compare_windows.py` runs the same experiment twice, changing only the window size (50 vs 1000), and prints the two results side by side. It is a quick way to see the effect of configuration on detection delay.

```bash
python demo_compare_windows.py
```

## Configuration

All experiment parameters live in [`02_implementation/config.yaml`](02_implementation/config.yaml): dataset paths and target columns, the feature that drift is injected into for each dataset, detector thresholds, the constraint grid, drift magnitude, and number of repetitions. The default random seed is 42.

## Datasets

| Dataset | Task | Drift feature |
|---|---|---|
| [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | Credit default prediction | `AGE` |
| IBM Telco Customer Churn | Customer churn prediction | `Monthly Charges` |
| [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | Machine failure prediction | `Tool wear [min]` |

## Limitations

- Drift is simulated, which allows exact delay measurement but may not reflect every pattern seen in real deployments.
- Detection is univariate, on one primary feature per dataset.
- Detector thresholds are fixed rather than tuned per configuration.
- The study covers detection only, not model retraining or adaptation.

See Chapter 5 of the thesis in `03_thesis/chapters/` for the full discussion.
