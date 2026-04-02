# FYP Project Structure - Modular Design

```
FYP/
├── 📋 Project Documentation
│   ├── proposal.md                 # Original research proposal
│   ├── outline.md                  # Detailed project roadmap
│   ├── PROJECT_STRUCTURE.md        # This file
│   └── EXECUTION_PLAN.md           # Step-by-step execution guide
│
├── 📚 01_literature_review/        # Phase 1: Research foundation
│   ├── README.md                   # Literature review guide
│   ├── systematic_review_template.md
│   ├── papers/                     # Downloaded papers (organized by topic)
│   │   ├── drift_detection/
│   │   ├── resource_constraints/
│   │   ├── mlops_systems/
│   │   └── edge_ml/
│   └── notes/                      # Your synthesis notes
│       ├── drift_fundamentals.md
│       ├── detection_methods.md
│       ├── constraints_landscape.md
│       └── research_gaps.md
│
├── 🔬 02_implementation/           # Phase 2-4: Core experimental system
│   ├── requirements.txt            # Python dependencies
│   ├── config.yaml                 # Global configuration
│   ├── __init__.py
│   │
│   ├── 📊 data/                    # Dataset management
│   │   ├── __init__.py
│   │   ├── loader.py               # Dataset loading & preprocessing
│   │   ├── preprocessor.py         # Feature engineering
│   │   └── README.md               # Dataset documentation
│   │
│   ├── 🤖 models/                  # Model training & inference
│   │   ├── __init__.py
│   │   ├── trainer.py              # Model training pipeline
│   │   ├── predictor.py            # Inference wrapper
│   │   └── README.md               # Model registry
│   │
│   ├── 🎯 detectors/               # Drift detection implementations
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract DriftDetector class
│   │   ├── ks_test.py              # Kolmogorov-Smirnov detector
│   │   ├── psi.py                  # Population Stability Index
│   │   ├── wasserstein.py          # Wasserstein distance detector
│   │   └── README.md               # Detector documentation
│   │
│   ├── 🌊 simulators/              # Drift pattern simulation
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract DriftSimulator class
│   │   ├── drift_patterns.py       # Sudden, gradual, recurring, compound
│   │   └── README.md               # Simulation documentation
│   │
│   ├── ⚙️ experiments/              # Experimental runner & orchestration
│   │   ├── __init__.py
│   │   ├── runner.py               # Main experiment executor
│   │   ├── config_generator.py     # Generate experiment configurations
│   │   └── README.md               # Experiment guide
│   │
│   ├── 📈 evaluation/               # Metrics & analysis
│   │   ├── __init__.py
│   │   ├── metrics.py              # Detection delay, FPR, accuracy drop, etc.
│   │   ├── statistical_tests.py    # Hypothesis testing, significance
│   │   └── README.md               # Evaluation metrics documentation
│   │
│   └── 🛠️ utils/                    # Shared utilities
│       ├── __init__.py
│       ├── logger.py               # Logging configuration
│       ├── resource_monitor.py     # CPU/memory tracking
│       └── helpers.py              # Common functions
│
├── 📖 03_thesis/                   # Phase 5-6: Writing & compilation
│   ├── thesis_main.md              # Root thesis document
│   │
│   ├── chapters/                   # Thesis chapters (in order)
│   │   ├── 00_abstract.md
│   │   ├── 01_introduction.md
│   │   ├── 02_literature_review.md
│   │   ├── 03_methodology.md
│   │   ├── 04_experimental_design.md
│   │   ├── 05_implementation.md
│   │   ├── 06_results.md
│   │   ├── 07_discussion.md
│   │   ├── 08_conclusion.md
│   │   └── 09_references.md
│   │
│   ├── figures/                    # Generated thesis figures
│   ├── tables/                     # Generated thesis tables
│   └── README.md                   # Thesis writing guide
│
├── 📊 04_results/                  # Phase 5: Experiment outputs
│   ├── experiments/                # Raw experiment data (JSON/CSV)
│   │   ├── baseline_credit.json
│   │   ├── constrained_credit.json
│   │   └── ...
│   │
│   ├── analysis/                   # Processed analysis data
│   │   ├── detection_delays.csv
│   │   ├── false_positive_rates.csv
│   │   └── ...
│   │
│   └── plots/                      # Generated visualizations
│       ├── detection_delay_vs_frequency.png
│       ├── fpr_vs_window_size.png
│       └── ...
│
└── 📝 .gitignore                   # Git ignore rules
```

## Key Design Principles

### 1. **Modularity**
- Each module has a **single responsibility**
- Clear interfaces between modules (defined in `base.py` files)
- Import dependencies flow downward only

### 2. **Configuration-Driven**
- All experiment parameters in `config.yaml`
- No hardcoded values in code
- Easy to reproduce and modify experiments

### 3. **Reproducibility**
- Requirements.txt pinned to versions
- Experiment configs stored with results
- Random seeds controlled globally

### 4. **Testability**
- Each module can be tested independently
- Detectors tested on synthetic data
- Simulators tested for correctness

### 5. **Documentation**
- Each folder has README.md
- Code is self-documenting (clear variable names)
- Examples in docstrings

## Phase Mapping

| Phase | Folder | Duration | Output |
|-------|--------|----------|--------|
| Phase 1: Foundation | 01_literature_review | Months 1-2 | Literature review chapter |
| Phase 2: Implementation | 02_implementation | Months 3-4 | Working experimental system |
| Phase 3: Experimentation | 02_implementation + 04_results | Months 5-6 | Complete experimental results |
| Phase 4: Analysis | 04_results | Month 7 | Analysis chapter |
| Phase 5: Writing | 03_thesis | Month 8 | Complete draft thesis |
| Phase 6: Finalization | 03_thesis + all | Month 9 | Final thesis submission |
