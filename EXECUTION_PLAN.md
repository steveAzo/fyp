# EXECUTION PLAN: Month-by-Month + Immediate Actions

## Quick Answer: Which Dataset to Start With?

### **RECOMMENDED: Credit Card Default Prediction**

**Why this one?**
- ✅ **Covariate drift perfect fit**: Customer demographics naturally shift over time
- ✅ **Realistic scenarios**: Fraud detection systems need monitoring in real deployments
- ✅ **Available**: UCI ML Repository + Kaggle
- ✅ **Good size**: ~30,000 samples (enough for experiments)
- ✅ **Public**: No proprietary data issues
- ✅ **Features**: Age, income, credit history, payment history → drift-prone features
- ✅ **Practical relevance**: Banks actually use drift detection (you'll cite their PSI use)

**Alternatives ranked by priority**:
1. **Telco Customer Churn** (~7,000 samples) - Good, smaller, faster to prototype
2. **Predictive Maintenance** (~10,000) - IoT-relevant, justifies edge device context
3. **Bike Sharing Demand** (~17,000) - Weather features have natural drift

---

## Phase 0: IMMEDIATE ACTIONS (This Week)

### Task 1: Download & Explore Credit Default Dataset

**Where**: 
- Option A: UCI ML Repository: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Option B: Kaggle: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

**What to do**:
```python
# Quick exploration (5 minutes, run locally or in Colab)
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('credit_default.csv')

# Explore
print(df.shape)           # Should be ~30,000 rows
print(df.info())          # See feature types
print(df.describe())      # Basic stats
print(df['target'].value_counts())  # Class imbalance

# Save locally
df.to_csv('02_implementation/data/credit_raw.csv', index=False)
```

**Deliverable**: Raw data file in `02_implementation/data/credit_raw.csv`

---

### Task 2: Create Project Initialization Script

**File**: `02_implementation/setup.py`

```python
"""
Project initialization script.
Run this ONCE to set up environment and verify everything works.
"""

import os
import sys

# Verify folder structure
required_folders = [
    'data', 'models', 'detectors', 'simulators', 
    'experiments', 'evaluation', 'utils'
]

for folder in required_folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✓ {folder}/ exists")

# Verify Python version
if sys.version_info < (3, 8):
    print("❌ Python 3.8+ required")
    sys.exit(1)

print("\n✓ Project structure initialized")
print("\nNext steps:")
print("1. pip install -r requirements.txt")
print("2. Download dataset to data/credit_raw.csv")
print("3. python 02_implementation/experiments/01_baseline.py")
```

---

### Task 3: Create requirements.txt

**File**: `02_implementation/requirements.txt`

```txt
# Core data science
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# ML models
lightgbm==4.0.0

# Statistical tests
scipy==1.11.2

# Visualization (for results)
matplotlib==3.7.2
seaborn==0.12.2

# Utilities
pyyaml==6.0
tqdm==4.66.1

# Development (optional)
jupyter==1.0.0
black==23.9.1
pytest==7.4.0
```

**Run once**:
```bash
cd 02_implementation
pip install -r requirements.txt
```

---

## MONTH 1: Foundation & Literature Review

### Week 1: Literature Review Scope (5 hours)

**Goal**: Define what papers are relevant, create search strategy

**Checklist**:
- [ ] Read `01_literature_review/README.md` thoroughly
- [ ] Write down your 4-5 main search queries (e.g., "data drift detection", "concept drift machine learning")
- [ ] Identify databases: Google Scholar, arXiv, IEEE, ACM
- [ ] Create spreadsheet template for tracking papers

**Deliverable**: Search strategy document (1 page)

---

### Weeks 2-3: Systematic Search (15 hours)

**Goal**: Find ~50 relevant papers, screen to ~40

**Process**:
1. **Google Scholar Search**
   - Query: "data drift detection"
   - Filter: 2015-2025
   - Screen titles/abstracts
   - Keep: ~30 papers

2. **arXiv Search**
   - Category: cs.LG (Machine Learning)
   - Search: "drift detection" OR "distribution shift"
   - Screen: ~10 papers

3. **Conference Proceedings** (if available)
   - Look for papers from: MLSys, NeurIPS, ICML
   - Manual search: "monitoring" OR "resource-constrained"

**Deliverable**: `01_literature_review/papers/papers_tracking.csv`
```csv
Title,Authors,Year,Venue,URL,Relevant_To,Notes
"Detecting Data Drift...",Smith et al,2018,ICML,https://...,Detection Methods,Proposes DDM
...
```

---

### Week 4 - Month 2, Week 1: Deep Reading (20 hours)

**Goal**: Read top 30-35 papers deeply, extract knowledge

**For EACH paper**:
- Create file: `01_literature_review/notes/AUTHOR_YEAR.md`
- Extract: Problem | Solution | Evaluation | Limitations | Relevance to YOUR work
- Take notes in your own words (not copy-paste)

**Parallel action**: Start writing synthesis docs

**Deliverable**: 30+ paper summaries + 4 synthesis documents

---

### Month 2, Weeks 2-4: Literature Review Chapter (20 hours)

**Goal**: Write Chapter 2 first draft

**Process**:
1. Use synthesis documents
2. Write following structure from `03_thesis/README.md` (Chapter 2 section)
3. Include tables comparing methods
4. End each section with "motivation for our study"

**Deliverable**: `03_thesis/chapters/02_literature_review.md` (10-12 pages)

---

## MONTH 2-3: Transition & Early Implementation

### Parallel Activity: Introduction Chapter (Month 2)

**Goal**: Write Chapter 1 once you understand field

**Deliverable**: `03_thesis/chapters/01_introduction.md` (4-5 pages)

---

## MONTH 3: Implementation Phase Begins

### Task 1: Create Base Classes & Module Structure (Week 1, 5 hours)

**Files to create**:

**`02_implementation/__init__.py`**: Empty init file

**`02_implementation/utils/logger.py`**:
```python
import logging
import os

def setup_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
```

**`02_implementation/detectors/base.py`**:
```python
from abc import ABC, abstractmethod
import numpy as np

class DriftDetector(ABC):
    """Abstract base class for all drift detectors."""
    
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.is_drift = False
        self.score = None
    
    @abstractmethod
    def detect(self, reference_data, test_data, feature):
        """
        Detect drift between reference and test data.
        
        Args:
            reference_data: Reference distribution (numpy array or pandas Series)
            test_data: Test distribution (numpy array or pandas Series)
            feature: Feature name (for logging)
        
        Returns:
            is_drift (bool): True if drift detected
            score (float): Detector-specific score
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold})"
```

**`02_implementation/simulators/base.py`**:
```python
from abc import ABC, abstractmethod
import numpy as np

class DriftSimulator(ABC):
    """Abstract base class for drift simulators."""
    
    def __init__(self, data, random_state=42):
        self.data = data.copy()
        self.random_state = random_state
        np.random.seed(random_state)
    
    @abstractmethod
    def apply_drift(self, start_idx, feature, **kwargs):
        """
        Apply drift to data starting at start_idx.
        
        Args:
            start_idx: Sample index where drift begins
            feature: Feature name to apply drift to
            **kwargs: Drift-specific parameters
        
        Returns:
            Modified dataframe with drift applied
        """
        pass
```

**Deliverable**: Core classes in place, ready for implementations

---

### Task 2: Implement Dataset Loader (Week 1-2, 5 hours)

**`02_implementation/data/loader.py`**:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DatasetLoader:
    """Load, preprocess, and split datasets."""
    
    def __init__(self, dataset_name='credit'):
        self.dataset_name = dataset_name
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_credit_default(self):
        """Load credit default dataset from UCI."""
        filepath = 'data/credit_raw.csv'
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset not found at {filepath}. "
                f"Download from: "
                f"https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
            )
        
        df = pd.read_csv(filepath)
        
        # Basic preprocessing
        # (adapt based on actual dataset structure)
        if 'default' in df.columns:
            self.y = df['default']
            self.X = df.drop('default', axis=1)
        else:
            raise ValueError("Target column 'default' not found")
        
        return self
    
    def preprocess(self):
        """Standardize features."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_scaled, columns=self.X.columns)
        return self
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """Split into train and test."""
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self
    
    def get_training_data(self):
        """Return training data."""
        return self.X_train, self.y_train
    
    def get_test_data(self):
        """Return test data."""
        return self.X_test, self.y_test

# Usage
# loader = DatasetLoader()
# loader.load_credit_default().preprocess().train_test_split()
# X_train, y_train = loader.get_training_data()
```

**Deliverable**: Working data loader

---

### Task 3: Implement KS-Test Detector (Week 2, 4 hours)

**`02_implementation/detectors/ks_test.py`**:

```python
from scipy.stats import ks_2samp
from .base import DriftDetector
import numpy as np

class KSTestDetector(DriftDetector):
    """Kolmogorov-Smirnov test for drift detection."""
    
    def __init__(self, threshold=0.05):
        super().__init__(threshold)
    
    def detect(self, reference_data, test_data, feature=None):
        """
        Perform KS test.
        
        Args:
            reference_data: Reference distribution (array-like)
            test_data: Test distribution (array-like)
            feature: Feature name (for logging, optional)
        
        Returns:
            is_drift (bool)
            p_value (float)
        """
        # Handle pandas Series
        if hasattr(reference_data, 'values'):
            reference_data = reference_data.values
        if hasattr(test_data, 'values'):
            test_data = test_data.values
        
        # Remove NaN values
        reference_data = reference_data[~np.isnan(reference_data)]
        test_data = test_data[~np.isnan(test_data)]
        
        # Perform KS test
        statistic, p_value = ks_2samp(reference_data, test_data)
        
        # Detect drift
        is_drift = p_value < self.threshold
        
        return is_drift, p_value
```

**Deliverable**: Working KS detector

---

### Task 4: Implement PSI Detector (Week 2-3, 5 hours)

**`02_implementation/detectors/psi.py`**:

```python
import numpy as np
from .base import DriftDetector

class PSIDetector(DriftDetector):
    """Population Stability Index detector."""
    
    def __init__(self, threshold=0.1, n_bins=10):
        super().__init__(threshold)
        self.n_bins = n_bins
    
    def detect(self, reference_data, test_data, feature=None):
        """
        Calculate PSI.
        
        Args:
            reference_data: Reference distribution
            test_data: Test distribution
            feature: Feature name (optional)
        
        Returns:
            is_drift (bool)
            psi_score (float)
        """
        # Handle pandas
        if hasattr(reference_data, 'values'):
            reference_data = reference_data.values
        if hasattr(test_data, 'values'):
            test_data = test_data.values
        
        # Remove NaN
        reference_data = reference_data[~np.isnan(reference_data)]
        test_data = test_data[~np.isnan(test_data)]
        
        # Create bins from reference
        breakpoints = np.percentile(
            reference_data, 
            np.arange(0, 101, 100/self.n_bins)
        )
        
        # Calculate proportions
        ref_counts = np.histogram(reference_data, breakpoints)[0]
        test_counts = np.histogram(test_data, breakpoints)[0]
        
        ref_props = ref_counts / len(reference_data)
        test_props = test_counts / len(test_data)
        
        # Avoid log(0)
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        test_props = np.where(test_props == 0, 0.0001, test_props)
        
        # Calculate PSI
        psi = np.sum(
            (test_props - ref_props) * np.log(test_props / ref_props)
        )
        
        is_drift = psi > self.threshold
        
        return is_drift, psi
```

**Deliverable**: Working PSI detector

---

### Task 5: First Prototype Test (Week 3, 3 hours)

**`02_implementation/experiments/01_baseline.py`**:

```python
"""
First working prototype: Train model + detect drift on clean data.
Goal: Verify everything works before constraints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data.loader import DatasetLoader
from models.trainer import ModelTrainer
from detectors.ks_test import KSTestDetector
from detectors.psi import PSIDetector
from utils.logger import setup_logger

logger = setup_logger('baseline')

def run_baseline():
    logger.info("Starting baseline experiment...")
    
    # Load data
    logger.info("Loading credit default dataset...")
    loader = DatasetLoader()
    loader.load_credit_default().preprocess().train_test_split()
    X_train, y_train = loader.get_training_data()
    X_test, y_test = loader.get_test_data()
    
    # Train model
    logger.info("Training RandomForest...")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    baseline_acc = trainer.evaluate(X_test, y_test)
    logger.info(f"Baseline accuracy: {baseline_acc:.3f}")
    
    # Test detectors (should detect NO drift on test set)
    logger.info("Testing detectors on drift-free data...")
    ks_detector = KSTestDetector(threshold=0.05)
    psi_detector = PSIDetector(threshold=0.1)
    
    feature = X_train.columns[0]  # Test on first feature
    
    # Use first 100 training samples as reference
    reference = X_train[feature].iloc[:100]
    
    # Use next 100 samples as test (should be same distribution)
    test_data = X_train[feature].iloc[100:200]
    
    is_drift_ks, score_ks = ks_detector.detect(reference, test_data)
    is_drift_psi, score_psi = psi_detector.detect(reference, test_data)
    
    logger.info(f"KS-Test: drift={is_drift_ks}, p_value={score_ks:.4f}")
    logger.info(f"PSI: drift={is_drift_psi}, score={score_psi:.4f}")
    
    if is_drift_ks or is_drift_psi:
        logger.warning("False positive detected! Check baseline data quality.")
    else:
        logger.info("✓ No false positives (as expected)")
    
    logger.info("Baseline prototype complete!")

if __name__ == '__main__':
    run_baseline()
```

**Run this**:
```bash
cd 02_implementation
python experiments/01_baseline.py
```

**Expected output**:
```
2024-01-10 14:30:45,123 - baseline - INFO - Starting baseline experiment...
2024-01-10 14:30:45,200 - baseline - INFO - Loading credit default dataset...
2024-01-10 14:30:46,100 - baseline - INFO - Training RandomForest...
2024-01-10 14:30:50,500 - baseline - INFO - Baseline accuracy: 0.823
2024-01-10 14:30:50,600 - baseline - INFO - Testing detectors on drift-free data...
2024-01-10 14:30:50,700 - baseline - INFO - KS-Test: drift=False, p_value=0.8234
2024-01-10 14:30:50,750 - baseline - INFO - PSI: drift=False, score=0.0032
2024-01-10 14:30:50,800 - baseline - INFO - ✓ No false positives (as expected)
```

**Deliverable**: Working prototype that:
- ✅ Loads data
- ✅ Trains model
- ✅ Detects no drift in clean data
- ✅ All modules connected

---

## MONTH 3-4: Full Implementation

### Remaining Tasks (Detailed in Phase 2):
- [ ] Implement Wasserstein detector
- [ ] Implement drift simulators (sudden, gradual, etc.)
- [ ] Implement experiment runner
- [ ] Implement evaluation metrics
- [ ] Create config.yaml for experiment parameters
- [ ] Write Chapter 3 & 4 (Research problem & Methods)

---

## MONTH 5-6: Experimentation

### Run Core Experiments:
- [ ] Baseline (unconstrained): 50 runs
- [ ] Frequency constraint: 100 runs
- [ ] Window size constraint: 100 runs
- [ ] Memory constraint: 100 runs
- [ ] Cross-dataset validation: 100 runs

**Total**: ~450 experiments × 10 repetitions = 4,500 runs

**Timeline**: Start early, run in batches

---

## MONTH 7: Analysis

### Analysis Tasks:
- [ ] Aggregate results from 4,500 runs
- [ ] Calculate statistics (mean, std, confidence intervals)
- [ ] Generate plots (detection delay vs frequency, etc.)
- [ ] Perform hypothesis tests
- [ ] Write Chapter 6 (Results)

---

## MONTH 8: Writing

### Writing Tasks:
- [ ] Draft Chapter 7 (Discussion)
- [ ] Draft Chapter 8 (Conclusion)
- [ ] Write Chapter 5 (Implementation details)
- [ ] Create abstract
- [ ] Full revision pass

---

## MONTH 9: Finalization

### Final Tasks:
- [ ] Address advisor feedback
- [ ] Final proofreading
- [ ] Generate final figures/tables
- [ ] Submit thesis

---

## Immediate Next Steps (This Week)

### Priority Order:
1. **TODAY**: Download credit default dataset, save to `02_implementation/data/credit_raw.csv`
2. **TODAY**: Create `requirements.txt`, run `pip install -r requirements.txt`
3. **TOMORROW**: Create base.py files for detectors and simulators
4. **THIS WEEK**: Implement KS and PSI detectors
5. **THIS WEEK**: Run 01_baseline.py to verify everything works
6. **NEXT WEEK**: Start literature review with search strategy

---

## Success Criteria for Each Phase

### Phase 1 (Lit Review): ✅ Success When...
- You've read 30-40 relevant papers
- You can explain drift detection to a non-ML person
- You've identified the research gap clearly
- Chapter 2 is written and coherent

### Phase 2 (Implementation): ✅ Success When...
- 01_baseline.py runs without errors
- All 3 detectors implemented and working
- Drift simulator produces detectable drift patterns
- Experiment runner can execute 10 configurations end-to-end

### Phase 3 (Experiments): ✅ Success When...
- 2,000+ experiments completed
- Results saved with consistent format
- Early analysis shows interesting patterns (not random noise)

### Phase 4 (Analysis): ✅ Success When...
- Chapter 6 written with clear findings
- Visualizations support claims
- Statistical tests show significance

### Phase 5 (Writing): ✅ Success When...
- Full draft complete
- All chapters coherent
- No orphaned sections

### Phase 6 (Finalization): ✅ Success When...
- Advisor happy with content
- Grammar/formatting perfect
- Submitted on deadline

