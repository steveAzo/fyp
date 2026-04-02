# QUICK START: Get Running in 1 Hour

## What You Need to Do RIGHT NOW

### Step 1: Download the Dataset (10 minutes)

**Option A (Recommended - Easier)**: Kaggle
1. Go to: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
2. Click "Download" (requires Kaggle account, free)
3. Extract the CSV file
4. Copy to: `c:\Users\User\Desktop\FYP\02_implementation\data\credit_raw.csv`

**Option B**: UCI Repository
1. Go to: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
2. Download the data file
3. Unzip and copy CSV to `02_implementation/data/credit_raw.csv`

**Verify** (in PowerShell):
```powershell
dir "C:\Users\User\Desktop\FYP\02_implementation\data\"
# Should see: credit_raw.csv (size ~10MB)
```

---

### Step 2: Install Python Dependencies (10 minutes)

```powershell
cd C:\Users\User\Desktop\FYP\02_implementation
pip install -r requirements.txt
```

**Expected output** (at end):
```
Successfully installed numpy-1.24.3 pandas-2.0.3 scikit-learn-1.3.0 ...
```

**Verify**:
```powershell
python -c "import pandas; import sklearn; import scipy; print('✓ All dependencies OK')"
```

---

### Step 3: Create Model Trainer (10 minutes)

Create file: `C:\Users\User\Desktop\FYP\02_implementation\models\trainer.py`

```python
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class ModelTrainer:
    """Train and evaluate ML models."""
    
    def __init__(self, model_type='rf'):
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1  # Use all cores
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train model."""
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set, return accuracy."""
        accuracy = self.model.score(X_test, y_test)
        return accuracy
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath):
        """Load saved model."""
        return joblib.load(filepath)
```

Create file: `C:\Users\User\Desktop\FYP\02_implementation\models\__init__.py` (empty)

---

### Step 4: Create Utils & Logger (5 minutes)

Create file: `C:\Users\User\Desktop\FYP\02_implementation\utils\__init__.py` (empty)

Then use the logger code from EXECUTION_PLAN.md, create file: `C:\Users\User\Desktop\FYP\02_implementation\utils\logger.py`

---

### Step 5: Create Detectors (15 minutes)

Create file: `C:\Users\User\Desktop\FYP\02_implementation\detectors\__init__.py` (empty)

Create three detector files:
- `base.py` (use code from EXECUTION_PLAN.md)
- `ks_test.py` (use code from EXECUTION_PLAN.md)
- `psi.py` (use code from EXECUTION_PLAN.md)

---

### Step 6: Create Data Loader (5 minutes)

Create file: `C:\Users\User\Desktop\FYP\02_implementation\data\__init__.py` (empty)

Create file: `C:\Users\User\Desktop\FYP\02_implementation\data\loader.py` with code from EXECUTION_PLAN.md

---

### Step 7: Run First Prototype (5 minutes)

Create file: `C:\Users\User\Desktop\FYP\02_implementation\experiments\01_baseline.py` with code from EXECUTION_PLAN.md

Then create: `C:\Users\User\Desktop\FYP\02_implementation\experiments\__init__.py` (empty)

Run it:
```powershell
cd C:\Users\User\Desktop\FYP\02_implementation
python experiments/01_baseline.py
```

**You should see**:
```
2024-01-10 14:30:45,123 - baseline - INFO - Starting baseline experiment...
...
2024-01-10 14:30:50,800 - baseline - INFO - ✓ No false positives (as expected)
```

---

## What This Achieves

✅ **Full modular structure set up**  
✅ **Data loading working**  
✅ **Model training working**  
✅ **KS & PSI detectors working**  
✅ **End-to-end pipeline verified**  

**You're now ready to**:
1. Start literature review (Month 1)
2. Expand with Wasserstein detector
3. Add drift simulators
4. Run full experiments

---

## Files Created So Far

```
FYP/
├── PROJECT_STRUCTURE.md           ← Read this first
├── EXECUTION_PLAN.md              ← Read this second  
├── 01_literature_review/
│   └── README.md                  ← Lit review guide (70 hours)
├── 02_implementation/
│   ├── requirements.txt           ← Dependencies
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── credit_raw.csv         ← Download here
│   ├── models/
│   │   ├── __init__.py
│   │   └── trainer.py             ← Train models
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ks_test.py
│   │   └── psi.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── experiments/
│       ├── __init__.py
│       └── 01_baseline.py         ← Run this first!
└── 03_thesis/
    ├── README.md                  ← Thesis chapter guide
    └── chapters/                  ← Write chapters here
```

---

## Next Actions (After Getting First Prototype Running)

### This Week:
1. ✅ Download dataset + verify
2. ✅ Run 01_baseline.py successfully
3. 📚 Read `01_literature_review/README.md`
4. 📝 Create literature review search strategy

### Next Week:
1. 📚 Start systematic paper search
2. 🔬 Implement Wasserstein detector (parallel to lit review)
3. 🔄 Implement drift simulators

### Week After:
1. 📚 Continue deep reading (20-30 papers)
2. 💻 Implement experiment runner
3. 🧪 Test with synthetic drift patterns

---

## Common Issues & Fixes

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
**Fix**: `pip install scikit-learn`

**Issue**: Dataset file not found
**Fix**: Check path is `02_implementation/data/credit_raw.csv`

**Issue**: KS detector returns strange p-values
**Fix**: Check for NaN values in data (loader should handle this)

**Issue**: Everything runs but results don't make sense
**Fix**: Add print statements, check baseline (should show NO drift on same distribution)

---

## Your Reading Priority

1. **TODAY**: This quick start guide
2. **TOMORROW**: `PROJECT_STRUCTURE.md` (understand the layout)
3. **THIS WEEK**: `EXECUTION_PLAN.md` (understand the timeline)
4. **THIS WEEK**: `01_literature_review/README.md` (plan your lit review)
5. **THIS WEEK**: `03_thesis/README.md` (understand thesis structure)

These documents ARE your project roadmap. Don't skip them.

---

## How to Get Help

### If you're confused about:
- **Repository structure**: See `PROJECT_STRUCTURE.md`
- **Timeline and tasks**: See `EXECUTION_PLAN.md`
- **Literature review process**: See `01_literature_review/README.md`
- **Thesis chapters**: See `03_thesis/README.md`
- **Code implementation**: Each module folder has a detailed example

### If code isn't working:
1. Read error message carefully
2. Check if you followed setup steps correctly
3. Verify dependencies installed: `pip list | grep -i pandas`
4. Add print statements to understand what's happening

### If you're stuck:
1. Check the relevant README.md file
2. Re-read the EXECUTION_PLAN.md timeline section
3. Ask your advisor (they've seen this before)

---

## One More Thing

**This structure is NOT final.** As you work, you'll:
- Add new modules (feature extraction, advanced analysis)
- Refactor old ones (simplify, improve)
- Remove unused code (keep it clean)

**Keep the structure clean through the project**. Future you will thank you.

Good luck! 🚀

