# INTERNAL PROJECT ROADMAP & DETAILED GUIDE

## When, Why, and Under What Constraints Do Lightweight Drift Detection Approaches Fail or Succeed?
### A Complete Guide to Your Final Year Project

---

## TABLE OF CONTENTS
1. [Project Overview & Philosophy](#1-project-overview--philosophy)
2. [Deep Dive: Understanding Core Concepts](#2-deep-dive-understanding-core-concepts)
3. [Technical Implementation Plan](#3-technical-implementation-plan)
4. [Experimental Design Details](#experimental-design-details)
5. [Analysis Framework](#analysis-framework)
6. [Avoiding the "Routine" Trap](#avoiding-the-routine-trap)
7. [Month-by-Month Execution Plan](#month-by-month-execution-plan)
8. [Publication Strategy](#publication-strategy)
9. [Common Pitfalls & How to Avoid Them](#common-pitfalls--how-to-avoid-them)
10. [Resources & References](#resources--references)

---

## 1. Project Overview & Philosophy

### 1.1 What Makes This Project Research-Grade?
**You're NOT building a product.** **You're NOT creating a new algorithm.** **You ARE conducting an empirical study that generates new knowledge** about when and why systems fail.

**The transformation:**

- ❌ **Engineering mindset:** "I'll build a lightweight drift detector"
- ✅ **Research mindset:** "I'll systematically characterize the conditions under which drift detection fails"

**Why this matters:**
- Engineering projects get graded on "does it work?"
- Research projects get graded on "what did we learn?"
- Publications require the latter

### 1.2 Your Core Value Proposition
**The gap you're filling:**
Most drift detection research assumes:
- Unlimited computational resources
- Ability to monitor continuously
- Access to full historical data
- Cloud infrastructure

**Reality for many deployments:**
- Edge devices (IoT, mobile)
- Small organizations (limited budgets)
- Developing regions (unreliable infrastructure)
- Academic settings (no cloud credits)

**Your contribution:** Empirical evidence of what happens when we remove those assumptions.

### 1.3 Success Metrics

**Academic Success:**
- ✅ Systematic experimental evaluation
- ✅ Statistical rigor (confidence intervals, significance tests)
- ✅ Generalizable findings (multiple datasets)
- ✅ Deep analysis explaining "why" behind patterns

**Research Success:**
- ✅ Novel insights about failure conditions
- ✅ Actionable practitioner guidelines
- ✅ Foundation for future research
- ✅ Conference/workshop paper publication

**Personal Success:**
- ✅ Deep understanding of ML systems
- ✅ Experience with empirical research
- ✅ Portfolio piece for job applications
- ✅ Strong final year grade

---

## 2. Deep Dive: Understanding Core Concepts

### 2.1 Data Drift: The Problem
**Definition:** Change in the statistical properties of input data over time.

**Mathematical Framing:**
- Training data: P_train(X, Y)
- Deployment data at time t: P_t(X, Y)
- Drift occurs when: P_train ≠ P_t

**Types of Drift:**

**A. Covariate Drift (P(X) changes, P(Y|X) stays same)**
- Example: Customer age distribution shifts from 30-40 to 50-60
- Model assumptions still valid, but operating on different input distribution
- Most common in practice

**B. Concept Drift (P(Y|X) changes)**
- Example: Relationship between "number of login attempts" and "fraud" changes because hackers adapt
- Model assumptions break down
- Requires model retraining

**C. Label Drift (P(Y) changes)**
- Example: Class imbalance shifts (fraud rate goes from 1% to 5%)
- May or may not require retraining

*For your project: Focus primarily on covariate drift (most common, easier to simulate and detect).*

### 2.2 Drift Detection Methods

#### **Method 1: Kolmogorov-Smirnov Test (KS-Test)**
**What it does:** Compares two distributions to see if they're significantly different.

**How it works:**
1. Take reference distribution (training data for feature X)
2. Take test distribution (recent data for feature X)
3. Compute maximum distance between cumulative distributions
4. Return p-value (if p < 0.05, distributions are different = drift)

**Pros:**
- Non-parametric (no assumptions about distribution shape)
- Well-established, interpretable
- Works for univariate features

**Cons:**
- Univariate only (must test each feature separately)
- Sensitive to sample size
- Requires storing reference distribution (memory cost)

**When it fails under constraints:**
- Low monitoring frequency: May miss drift between checks
- Small window size: High variance in p-values, many false positives
- Limited memory: Can't maintain sufficient reference samples

**Implementation:**
```python
from scipy.stats import ks_2samp

# Reference data (training)
reference = training_data['age']

# Test data (recent batch)
test = current_batch['age']

# Perform test
statistic, p_value = ks_2samp(reference, test)

if p_value < 0.05:
    print("Drift detected!")
```

#### **Method 2: Population Stability Index (PSI)**
**What it does:** Measures shift in distribution by comparing proportions across bins.

**How it works:**
1. Bin reference distribution into deciles (10 bins)
2. Calculate proportion in each bin
3. For new data, calculate proportion in same bins
4. Compute PSI = Σ (actual% - expected%) × ln(actual% / expected%)

**Interpretation:**
- PSI < 0.1: No significant change
- PSI 0.1-0.25: Moderate change, investigate
- PSI > 0.25: Significant shift, likely drift

**Pros:**
- Widely used in banking/finance (proven track record)
- Single number output (easy to threshold)
- Less sensitive to small sample sizes than KS-test
- Lower memory requirements (just store bin boundaries and counts)

**Cons:**
- Requires binning decisions (affects sensitivity)
- Can miss drift if it happens within bins
- Assumes categories/bins make sense for your data

**When it works better under constraints:**
- Memory-limited: Only need to store bin counts, not full distribution
- Interpretability: Single score easier to track over time

**Implementation:**
```python
import numpy as np

def calculate_psi(expected, actual, bins=10):
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.arange(0, 100, 100/bins))
    
    # Calculate proportions
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi
```

#### **Method 3: Wasserstein Distance (Earth Mover's Distance)**
**What it does:** Measures minimum "work" needed to transform one distribution into another.

**Intuition:** If you think of distributions as piles of dirt, how much dirt do you need to move (and how far) to make them identical?

**Pros:**
- Captures distribution shape well
- Continuous measure (not just binary drift/no-drift)
- Effective for gradual drift

**Cons:**
- More computationally expensive than KS or PSI
- Less interpretable than PSI
- Can be sensitive to outliers

**When it fails under constraints:**
- CPU-limited: Computation cost scales with sample size
- May be overkill for simple drift patterns

**Implementation:**
```python
from scipy.stats import wasserstein_distance

distance = wasserstein_distance(reference, test)

# You need to define a threshold (dataset-dependent)
if distance > threshold:
    print("Drift detected!")
```

### 2.3 Resource Constraints: What You're Actually Limiting

#### **Constraint 1: Monitoring Frequency**
**What it means:** How often you check for drift.
- **Baseline:** Check every 10-100 samples
- **Constrained:** Check every 500-1000 samples

**Why this matters:**
- High frequency: Catch drift quickly, but expensive (lots of compute/checks)
- Low frequency: Miss drift, model degrades longer before detection

**Real-world analogy:**
- Checking blood pressure every hour vs once a month
- Monthly checks might miss a sudden spike that causes problems

**Your experiment:** Vary frequency from every-10 to every-1000 samples, measure detection delay.

#### **Constraint 2: Window Size**
**What it means:** How much recent data you use for each drift check.
- **Baseline:** Last 1000-5000 samples
- **Constrained:** Last 50-100 samples

**Why this matters:**
- Large window: More stable statistics, better for gradual drift, but higher memory
- Small window: Fast to compute, responsive to sudden drift, but noisy (false positives)

**Real-world analogy:**
- Weather forecast using last 30 days vs last 3 days
- 3-day window responds quickly to changes but might overreact to outliers

**Your experiment:** Vary window size from 50 to 5000 samples, measure false positive rate and detection accuracy.

#### **Constraint 3: Memory Budget**
**What it means:** Maximum RAM available for storing reference data and statistics.
- **Baseline:** Unlimited (store full reference distribution)
- **Constrained:** 10MB, 50MB, 100MB

**Why this matters:**
- Full reference: Best detection accuracy, can store all training data
- Limited memory: Must subsample or use summary statistics, loses information

**Real-world example:**
- IoT device with 50MB RAM can't store 100,000 training samples
- Must either: (a) subsample to 1000 samples, (b) store only summary stats (mean, std, bins)

**Your experiment:** Artificially limit memory, force detectors to use smaller reference sets, measure impact on detection.

**Implementation approach:**
```python
# Baseline: Store all reference data
reference_full = training_data['age']  # 100,000 samples

# Constrained: Subsample to fit memory budget
memory_limit_mb = 10
sample_size = calculate_max_samples(memory_limit_mb)
reference_constrained = training_data['age'].sample(sample_size)  # 1,000 samples
```

#### **Constraint 4: Computational Budget**
**What it means:** Maximum CPU time allowed per drift check.

**Why this matters:**
- Some detectors are O(n²) complexity, too slow for large windows
- Edge devices have limited processing power

**Your experiment:** Measure actual CPU time for each detector, simulate timeouts.

### 2.4 Drift Patterns You'll Simulate

#### **Pattern 1: Sudden Drift**
**Description:** Abrupt change at a specific point in time.

**Real-world examples:**
- Policy change (sudden change in customer demographics)
- System update (new data source with different characteristics)
- Market crash (financial data shifts overnight)

**Simulation:**
```python
# Original data: mean=50, std=10
data_before = np.random.normal(50, 10, 1000)

# Sudden drift at sample 1000: mean=70, std=10
data_after = np.random.normal(70, 10, 1000)

data_stream = np.concatenate([data_before, data_after])
```

**Expected behavior:**
- Should be easiest to detect
- Even low-frequency monitoring should catch it
- Test: How quickly after sample 1000 is drift detected?

#### **Pattern 2: Gradual Drift**
**Description:** Slow, continuous change over time.

**Real-world examples:**
- Aging population (customer age slowly increases)
- Climate change (temperature trends)
- Product lifecycle (user behavior evolves slowly)

**Simulation:**
```python
# Gradually shift mean from 50 to 70 over 5000 samples
samples = 5000
means = np.linspace(50, 70, samples)
data_stream = [np.random.normal(mean, 10, 1) for mean in means]
```

**Expected behavior:**
- Hardest to detect (change is subtle at each step)
- Requires large windows and frequent monitoring
- Test: At what sample does drift become detectable?

#### **Pattern 3: Recurring/Seasonal Drift**
**Description:** Periodic pattern that repeats.

**Real-world examples:**
- Seasonal shopping patterns (holiday spikes)
- Weekly cycles (weekday vs weekend behavior)
- Time-of-day patterns (morning vs evening traffic)

**Simulation:**
```python
# Sine wave drift: mean oscillates between 40 and 60
samples = 5000
t = np.linspace(0, 4*np.pi, samples)
means = 50 + 10 * np.sin(t)
data_stream = [np.random.normal(mean, 10, 1) for mean in means]
```

**Expected behavior:**
- Detectors may trigger repeatedly (false alarms if seasonality is expected)
- Requires drift detector to distinguish "normal seasonality" from "true drift"
- Test: Do detectors adapt to recurring patterns?

#### **Pattern 4: Compound Drift**
**Description:** Multiple drift types combined (gradual + sudden, etc.)

**Simulation:**
```python
# Gradual drift from 50 to 60 over 2000 samples
# Then sudden jump to 80 at sample 2000
gradual_means = np.linspace(50, 60, 2000)
sudden_mean = 80

data_gradual = [np.random.normal(mean, 10, 1) for mean in gradual_means]
data_sudden = np.random.normal(sudden_mean, 10, 3000)

data_stream = np.concatenate([data_gradual, data_sudden])
```

**Expected behavior:**
- Tests robustness of detectors to complex patterns
- Should detect at least the sudden component

#### **Pattern 5: Drift Magnitude Variations**
**Description:** Same drift type but different intensities.

**Subtle drift:** Mean shifts from 50 to 52 (small)  
**Severe drift:** Mean shifts from 50 to 100 (large)

**Why this matters:**
- Tests sensitivity limits of detectors
- Under resource constraints, can detectors still catch subtle drift?

---

## 3. Technical Implementation Plan

### 3.1 System Architecture

**Component Breakdown:**
```
Your System
├── Data Layer
│   ├── Dataset loader (multiple datasets)
│   └── Data preprocessing
├── Model Layer
│   ├── Model training (one-time)
│   └── Model inference (ongoing)
├── Drift Simulation Layer
│   ├── Drift pattern generators
│   └── Data stream simulator
├── Detection Layer
│   ├── KS-Test detector
│   ├── PSI detector
│   └── Wasserstein detector
├── Monitoring Layer
│   ├── Configuration manager (frequency, window, memory)
│   └── Resource tracker (CPU, memory usage)
├── Evaluation Layer
│   ├── Metrics calculator
│   └── Results logger
└── Analysis Layer
    ├── Statistical analysis
    └── Visualization generator
```

### 3.2 Technology Stack

**Core Libraries:**
```python
# Data manipulation
import pandas as pd
import numpy as np

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Drift detection
from scipy.stats import ks_2samp, wasserstein_distance

# Evaluation
from sklearn.metrics import accuracy_score, f1_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Resource monitoring
import time
import psutil  # Memory usage tracking
```

**No Need For:**
- ❌ Docker/Kubernetes (not deploying to production)
- ❌ MLflow/Weights & Biases (nice to have, not essential)
- ❌ Complex MLOps tools (defeats the purpose of studying constraints)

### 3.3 Dataset Selection

**Criteria:**
- ✅ Publicly available (Kaggle, UCI)
- ✅ Tabular data (easier to simulate drift)
- ✅ Classification or regression task
- ✅ Diverse domains (generalizability)
- ✅ Sufficient samples (10,000+)

**Recommended Datasets (Pick 3-4):**

1. **Credit Card Default (Classification)**
   - Domain: Finance
   - Features: Age, income, credit history
   - Target: Will default on loan?
   - Size: ~30,000 samples
   - Link: UCI ML Repository / Kaggle

2. **Predictive Maintenance (Classification)**
   - Domain: Manufacturing/IoT
   - Features: Temperature, vibration, pressure readings
   - Target: Will machine fail?
   - Size: ~10,000 samples
   - Link: Kaggle "Predictive Maintenance Dataset"

3. **Customer Churn (Classification)**
   - Domain: Business/Telecom
   - Features: Contract length, monthly charges, services
   - Target: Will customer leave?
   - Size: ~7,000 samples
   - Link: Kaggle "Telco Customer Churn"

4. **Bike Sharing Demand (Regression)**
   - Domain: Transportation
   - Features: Weather, time, season
   - Target: Number of rentals
   - Size: ~17,000 samples
   - Link: Kaggle "Bike Sharing Demand"

**Why multiple datasets?**
- Shows your findings generalize
- Different drift characteristics (e.g., weather data has natural seasonality)
- Strengthens publication case

### 3.4 Model Selection & Training

**Keep it SIMPLE:**
```python
# Example: Credit Default Dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('credit_default.csv')
X = data.drop('default', axis=1)
y = data['default']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Baseline performance
baseline_accuracy = model.score(X_test, y_test)
print(f"Baseline accuracy: {baseline_accuracy:.3f}")

# Save model
import joblib
joblib.dump(model, 'credit_model.pkl')
```

**Why Random Forest?**
- ✅ Works well out-of-box (less tuning needed)
- ✅ Handles mixed feature types
- ✅ Interpretable (feature importance)
- ✅ Fast inference

**Alternatives:**
- Logistic Regression (simpler, faster)
- LightGBM (more accurate, still lightweight)

**What you DON'T need:**
- ❌ Deep learning (overkill for tabular data)
- ❌ Extensive hyperparameter tuning (not the focus)
- ❌ Model comparison (you're evaluating drift detection, not models)

### 3.5 Drift Simulation Implementation

**Modular Design:**
```python
class DriftSimulator:
    def __init__(self, original_data, drift_config):
        self.original_data = original_data
        self.drift_config = drift_config
    
    def generate_stream(self):
        """Generate data stream with simulated drift"""
        pass
    
    def sudden_drift(self, start_sample, feature, shift):
        """Apply sudden shift to a feature at specific sample"""
        pass
    
    def gradual_drift(self, start_sample, end_sample, feature, shift):
        """Apply gradual drift over sample range"""
        pass
    
    def recurring_drift(self, feature, period, amplitude):
        """Apply periodic drift pattern"""
        pass
```

**Example Implementation:**
```python
def sudden_drift(data, start_sample, feature, shift_mean=20, shift_std=0):
    """
    Apply sudden drift to a feature.
    
    Args:
        data: Original dataframe
        start_sample: Sample index where drift begins
        feature: Column name to apply drift
        shift_mean: How much to shift the mean
        shift_std: How much to shift the std (optional)
    
    Returns:
        Modified dataframe with drift
    """
    data_copy = data.copy()
    
    # Apply shift only after start_sample
    data_copy.loc[start_sample:, feature] += shift_mean
    
    if shift_std != 0:
        data_copy.loc[start_sample:, feature] *= (1 + shift_std)
    
    return data_copy

# Usage
original_data = pd.read_csv('credit_default.csv')
drifted_data = sudden_drift(original_data, start_sample=5000, feature='age', shift_mean=10)
```

**Key Insight:** You control the "ground truth" of when drift happens, making evaluation clean.

### 3.6 Drift Detector Implementation

**Base Class:**
```python
class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_detected = False
        self.detection_time = None
    
    def detect(self, test_data):
        """
        Compare test data to reference, return drift signal.
        
        Returns:
            is_drift (bool), score (float)
        """
        raise NotImplementedError

class KSTestDetector(DriftDetector):
    def detect(self, test_data, feature):
        from scipy.stats import ks_2samp
        
        ref = self.reference_data[feature]
        test = test_data[feature]
        
        statistic, p_value = ks_2samp(ref, test)
        
        is_drift = p_value < self.threshold
        return is_drift, p_value

class PSIDetector(DriftDetector):
    def __init__(self, reference_data, threshold=0.1):
        super().__init__(reference_data, threshold)
        # Precompute reference bins
        self.bins = {}
    
    def detect(self, test_data, feature):
        psi_score = self.calculate_psi(
            self.reference_data[feature],
            test_data[feature]
        )
        
        is_drift = psi_score > self.threshold
        return is_drift, psi_score
    
    def calculate_psi(self, expected, actual, bins=10):
        # (Implementation from earlier)
        pass
```

### 3.7 Experimental Runner

**Core Loop:**
```python
class ExperimentRunner:
    def __init__(self, model, detector, config):
        self.model = model
        self.detector = detector
        self.config = config  # Monitoring frequency, window size, etc.
        self.results = []
    
    def run_experiment(self, data_stream, drift_start_time):
        """
        Simulate deployment with drift detection.
        
        Args:
            data_stream: Full dataset with simulated drift
            drift_start_time: Sample index where drift begins (ground truth)
        
        """
        window_size = self.config['window_size']
        check_frequency = self.config['check_frequency']

        detection_time = None
        false_positives = 0
    
    for i in range(0, len(data_stream), check_frequency):
        # Get current window
        window_start = max(0, i - window_size)
        current_window = data_stream[window_start:i]
        
        if len(current_window) < window_size:
            continue  # Not enough data yet
        
        # Check for drift
        is_drift, score = self.detector.detect(current_window)
        
        # Log result
        if is_drift:
            if detection_time is None:
                detection_time = i
                
            if i < drift_start_time:
                false_positives += 1
        
        # Measure model performance
        y_true = current_window['target']
        y_pred = self.model.predict(current_window.drop('target', axis=1))
        accuracy = accuracy_score(y_true, y_pred)
        
        # Log metrics
        self.results.append({
            'sample_index': i,
            'drift_detected': is_drift,
            'drift_score': score,
            'accuracy': accuracy,
            'true_drift_started': drift_start_time,
            'detection_delay': i - drift_start_time if is_drift and i >= drift_start_time else None
        })
    
        return {
            'detection_time': detection_time,
            'detection_delay': detection_time - drift_start_time if detection_time else None,
            'false_positives': false_positives,
            'final_accuracy': self.results[-1]['accuracy']
        }
```


---

### **3.8 Resource Monitoring**

**Track actual resource usage:**
```python
import psutil
import time

class ResourceMonitor:
    def __init__(self):
        self.cpu_times = []
        self.memory_usage = []
    
    def start_monitoring(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / (1024**2)  # MB
    
    def stop_monitoring(self):
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**2)
        
        cpu_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        return {
            'cpu_time_seconds': cpu_time,
            'memory_mb': memory_delta
        }

# Usage in experiment
monitor = ResourceMonitor()

monitor.start_monitoring()
is_drift, score = detector.detect(test_data)
resources = monitor.stop_monitoring()

print(f"Detection took {resources['cpu_time_seconds']:.4f}s and {resources['memory_mb']:.2f}MB")
```

---

## **4. Experimental Design Details**

### **4.1 Experimental Variables (Formal)**

**Independent Variables (What You Control):**

| Variable | Type | Levels | Description |
|----------|------|--------|-------------|
| Monitoring Frequency | Discrete | {10, 50, 100, 500, 1000} samples | How often drift check runs |
| Window Size | Discrete | {50, 100, 500, 1000, 5000} samples | Size of recent data window |
| Memory Budget | Discrete | {10, 50, 100, 500, unlimited} MB | Max reference data storage |
| Drift Type | Categorical | {sudden, gradual, recurring, compound} | Pattern of distribution change |
| Drift Magnitude | Continuous | {small, medium, large} | Intensity of shift |
| Detector Method | Categorical | {KS-Test, PSI, Wasserstein} | Algorithm used |
| Dataset | Categorical | {Credit, Maintenance, Churn, Bike} | Domain/task |

**Dependent Variables (What You Measure):**

| Variable | Type | Unit | Formula/Method |
|----------|------|------|----------------|
| Detection Delay | Continuous | Samples | t_detected - t_drift_start |
| False Positive Rate | Continuous | Ratio | FP / total_checks_before_drift |
| False Negative (Missed Drift) | Binary | Yes/No | Did detector never trigger after drift? |
| Model Accuracy Drop | Continuous | % | accuracy_before - accuracy_after |
| CPU Time per Check | Continuous | Seconds | Measured via timer |
| Memory Usage | Continuous | MB | Measured via psutil |
| Monitoring Overhead | Continuous | % | (drift_check_time / total_time) * 100 |

**Control Variables (Keep Constant):**
- Model architecture (same Random Forest across all experiments)
- Training data (same split for all experiments on a dataset)
- Random seeds (for reproducibility)
- Drift start point (e.g., always at sample 5000)

---

### **4.2 Experimental Configurations**

**Configuration Matrix (Example Subset):**

| Experiment ID | Frequency | Window | Memory | Drift Type | Dataset |
|---------------|-----------|--------|--------|-----------|---------|
| EXP-001 | 10 | 1000 | Unlimited | Sudden | Credit |
| EXP-002 | 10 | 1000 | Unlimited | Gradual | Credit |
| EXP-003 | 1000 | 1000 | Unlimited | Sudden | Credit |
| EXP-004 | 1000 | 1000 | Unlimited | Gradual | Credit |
| EXP-005 | 10 | 100 | Unlimited | Sudden | Credit |
| EXP-006 | 10 | 100 | Unlimited | Gradual | Credit |
| EXP-007 | 10 | 1000 | 50MB | Sudden | Credit |
| ... | ... | ... | ... | ... | ... |

**Full factorial design would be:**
- 5 frequencies × 5 window sizes × 5 memory levels × 4 drift types × 3 detectors × 4 datasets = **6,000 experiments**

**Practical reduction strategies:**

1. **Prioritize key factors:** Focus on frequency and window size first (most impactful)
2. **Fractional factorial:** Test key combinations, not all
3. **Iterative:** Start with 1 dataset, expand if time permits
4. **Baseline vs extremes:** Test unconstrained baseline, then extreme constraints

**Realistic experiment count: 200-500 core experiments**

Each experiment repeated 10 times = 2,000-5,000 runs total (still manageable, can run overnight).

---

### **4.3 Evaluation Metrics (Detailed)**

#### **Metric 1: Detection Delay**

**Definition:** Time elapsed between drift occurrence and detection.

**Formula:**

Detection_Delay = Sample_Index_Detected - Sample_Index_Drift_Started


**Interpretation:**
- Lower = Better (faster detection)
- Delay of 0-100 samples: Excellent
- Delay of 100-500 samples: Acceptable
- Delay > 1000 samples: Poor (model may degrade significantly)

**Why it matters:**
- During delay, model runs on drifted data → performance degrades
- Longer delay = more bad predictions = business impact

**Example:**
- Drift starts at sample 5000
- Detected at sample 5300
- Detection Delay = 300 samples

---

#### **Metric 2: False Positive Rate**

**Definition:** Proportion of drift alerts triggered when no drift exists.

**Formula:**

FPR = (False_Alarms / Total_Checks_Before_Drift)

**Interpretation:**
- Lower = Better (fewer false alarms)
- FPR < 0.05: Excellent (aligns with 0.05 p-value threshold)
- FPR > 0.20: Poor (too many false alarms)

**Why it matters:**
- False alarms cause unnecessary model retraining
- Wastes compute resources
- "Alert fatigue" → real drift might be ignored

**Example:**
- 100 drift checks before actual drift
- 3 triggered false alarms
- FPR = 3/100 = 0.03 (good)

---

#### **Metric 3: False Negative (Missed Drift)**

**Definition:** Binary indicator if drift was never detected.

**Formula:**


Missed_Drift = (Detection_Time == None) after sufficient time 
**Interpretation:**
- 0 = Detected (good)
- 1 = Missed (catastrophic failure)

**Why it matters:**
- Missed drift = model continues degrading indefinitely
- Worst possible outcome for monitoring system

---

#### **Metric 4: Model Performance Degradation**

**Definition:** Change in model accuracy from before drift to after drift (undetected period).

**Formula:**

Accuracy_Drop = Accuracy_Before_Drift - Accuracy_At_Detection


**Interpretation:**
- Lower = Better (less degradation)
- Drop < 5%: Minimal impact
- Drop 5-15%: Moderate impact
- Drop > 15%: Severe impact

**Why it matters:**
- Quantifies business impact of delayed detection
- Connects drift detection effectiveness to downstream consequences

**Example:**
- Accuracy before drift: 0.92
- Accuracy at detection: 0.85
- Accuracy Drop = 0.07 (7% degradation)

---

#### **Metric 5: Resource Consumption**

**CPU Time per Check:**
CPU_Time = Time_End - Time_Start  (seconds)

**Memory Usage:**
Memory_MB = Peak_Memory - Baseline_Memory  (megabytes)


**Monitoring Overhead:**

Overhead_% = (Total_Monitoring_Time / Total_Execution_Time) * 100

**Why it matters:**
- Direct measure of "cost" of monitoring
- Validates whether constraints are truly limiting

---

### **4.4 Statistical Analysis Plan**

**Goal:** Ensure findings are statistically significant, not due to random chance.

#### **Step 1: Descriptive Statistics**

For each metric, report:
- Mean
- Median
- Standard deviation
- Min/Max
- Confidence intervals (95%)

**Example:**


Detection Delay (Baseline Configuration):
Mean: 120 samples
Median: 110 samples
Std Dev: 25 samples
95% CI: [115, 125]

---

#### **Step 2: Hypothesis Testing**

**Null Hypothesis (H0):** Resource constraints have no effect on detection delay.

**Alternative (H1):** Resource constraints increase detection delay.

**Test:** Paired t-test or Wilcoxon signed-rank test
```python
from scipy.stats import ttest_rel

# Detection delays for baseline vs constrained (same drift instances)
baseline_delays = [120, 115, 130, ...]
constrained_delays = [450, 430, 480, ...]

statistic, p_value = ttest_rel(baseline_delays, constrained_delays)

if p_value < 0.05:
    print("Constraint effect is statistically significant")
```

---

#### **Step 3: Effect Size**

Report Cohen's d to quantify magnitude of difference:
```python
def cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return mean_diff / pooled_std

effect = cohens_d(baseline_delays, constrained_delays)
# effect > 0.8 = large effect
# effect 0.5-0.8 = medium
# effect < 0.5 = small
```

---

#### **Step 4: Regression Analysis**

Model detection delay as function of constraints:
```python
import statsmodels.api as sm

# Prepare data
X = pd.DataFrame({
    'monitoring_frequency': [...],
    'window_size': [...],
    'memory_budget': [...]
})
y = detection_delays

# Fit regression
model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())

# Interpretation: Which constraint has largest coefficient?
```

**This tells you:** "Increasing monitoring frequency by 100 samples increases detection delay by X samples (holding window size constant)."

---

#### **Step 5: Interaction Effects**

Test if constraints interact (e.g., low frequency + small window is worse than sum of individual effects):
```python
# Add interaction term
X['freq_x_window'] = X['monitoring_frequency'] * X['window_size']

model_interaction = sm.OLS(y, sm.add_constant(X)).fit()
```

---

### **4.5 Ablation Studies**

**Purpose:** Isolate effect of individual constraint dimensions.

**Experiment Design:**

**Baseline:** All resources unlimited  
**Ablation 1:** Only frequency constrained  
**Ablation 2:** Only window constrained  
**Ablation 3:** Only memory constrained  
**Ablation 4:** All constrained (compound effect)

**Analysis:**
- If Ablation 4 effect ≈ Sum(Ablation 1 + 2 + 3): Constraints are additive
- If Ablation 4 effect >> Sum: Constraints interact negatively (synergistic failure)

---

## **5. Analysis Framework**

### **5.1 From Data to Insights**

**Your analysis should answer:**

1. **Which constraints hurt the most?**
   - "Reducing monitoring frequency has 3x the impact of reducing window size."

2. **Which drift types are most vulnerable?**
   - "Gradual drift detection fails completely under memory constraints."

3. **Are there threshold effects?**
   - "Below 100 samples per check, all detectors fail."

4. **Do methods respond differently?**
   - "PSI is more robust to memory constraints than KS-Test."

5. **Can we predict failure?**
   - "If frequency > 500 AND drift is gradual, expect detection delay > 1000 samples."

---

### **5.2 Failure Taxonomy**

**Categorize failures into types:**

| Failure Mode | Definition | Cause | Example Configuration |
|--------------|-----------|-------|----------------------|
| **Complete Miss** | Drift never detected | Insufficient sensitivity | Frequency = 1000, Window = 50 |
| **Late Detection** | Detected but with long delay | Slow response to gradual changes | Frequency = 500, Gradual drift |
| **False Alarm Cascade** | Many false positives | High variance due to small window | Window = 50, noisy data |
| **Degraded Accuracy** | Detection occurs but model already degraded >10% | Combination of delay + drift severity | Memory = 10MB, severe drift |

**For each failure mode:**
- Describe characteristic patterns
- Identify triggering conditions
- Propose mitigation strategies

---

### **5.3 Trade-off Curves**

**Create visualizations showing:**

**1. Accuracy-Resource Trade-off:**
   Detection Accuracy (%)
   ^
100% |  •
|    •
90% |      •
|         •
80% |              •
|                     •___
70% |_________________________•
+-----|-----|-----|--------|->
10MB  50MB  100MB  500MB  Memory Budget

**Insight:** "To maintain 90% detection accuracy, you need at least 50MB memory."

---

**2. Delay-Frequency Relationship:**
   Detection Delay (samples)
   ^
1000  |                        /
|                      /
500  |                    /
|                  /
100  |            /
|   /
50  |__/___________
+-----|-----|-----|--------|->
10    100   500   1000  Monitoring Frequency

**Insight:** "Delay increases exponentially beyond 500-sample frequency."

---

**3. Method Comparison Under Constraints:**
   Detection Delay (samples)
   ^
800  |                    PSI ----
|                 /
600  |              /
|           /  KS-Test ----
400  |        /
|     /
200  |  /
|/________________________
+-----|-----|-----|--------|->
Baseline  Low    Med   High  Constraint Level

**Insight:** "PSI degrades more gracefully than KS-Test under constraints."

---

### **5.4 Practitioner Guidelines**

**Deliverable:** Decision tree or table for deployment decisions.

**Example Decision Tree:**
Is deployment edge device with <50MB memory?
├─ YES: Use PSI detector
│   ├─ Is drift expected to be gradual?
│   │   ├─ YES: Monitor every 100 samples minimum
│   │   └─ NO: Can monitor every 500 samples
│   └─ Set PSI threshold to 0.2 (more sensitive)
└─ NO (Cloud/server deployment):
└─ Use KS-Test
├─ Monitor every 50 samples for best accuracy
└─ Window size = 1000 samples


**Example Table:**

| Deployment Scenario | Recommended Method | Min Monitoring Frequency | Min Window Size | Expected Detection Delay |
|---------------------|-------------------|-------------------------|----------------|-------------------------|
| Edge device (<50MB) | PSI | Every 100 samples | 100 samples | 200-400 samples |
| Small server (100-500MB) | KS-Test | Every 50 samples | 500 samples | 100-200 samples |
| Cloud (unlimited) | Wasserstein | Every 10 samples | 1000 samples | 50-100 samples |
| IoT sensor (extreme constraint) | PSI | Every 500 samples | 50 samples | 500-1000 samples |

---

### **5.5 Deep Analysis: The "Why" Chapter**

**This is what separates A from B grade.**

**For each major finding, explain the mechanism:**

**Example Finding:**
> "KS-Test detection delay increased by 400% when memory was limited to 10MB."

**Shallow analysis (bad):**
> "The constraint hurt performance."

**Deep analysis (good):**
> "The KS-Test requires maintaining the full reference distribution to compute the cumulative distribution function accurately. Under 10MB memory constraints, we were forced to subsample the reference data from 10,000 samples to approximately 1,500 samples. This subsampling introduced two effects: (1) loss of tail distribution information, making the test less sensitive to distributional changes in rare values, and (2) increased variance in the test statistic due to smaller sample size. Analysis of the p-values over time (Figure X) shows that the test statistic fluctuated more wildly under constraints, requiring more samples to accumulate evidence of drift before crossing the significance threshold. This explains the observed 4x increase in detection delay. The effect was particularly pronounced for gradual drift (Figure Y), where the cumulative evidence accumulation process is critical."

**Components of deep analysis:**
1. **Mechanism:** Why did this happen? (root cause)
2. **Evidence:** What in the data supports this explanation?
3. **Implications:** What does this mean for practitioners?

---

## **6. Avoiding the "Routine" Trap**

### **6.1 What Makes a Project Routine?**

**Routine characteristics:**
- ❌ "I built X and it works"
- ❌ Only reports whether something worked or not
- ❌ No systematic evaluation
- ❌ No insights beyond the implementation
- ❌ Could be replicated by following a tutorial

**Your project AVOIDS this by:**
- ✅ Asking "when and why" not just "does it work"
- ✅ Systematic experimental design
- ✅ Statistical rigor and multiple datasets
- ✅ Deep analysis explaining observed patterns
- ✅ Generalizable insights for practitioners

---

### **6.2 Adding Non-Routine Elements (Elevating the Work)**

**Recall my suggestions from earlier. Here's how to integrate them:**

#### **Option 1: Resource-Adaptive Meta-Strategy**

**Basic idea:** Dynamically switch detection methods based on available resources.

**Implementation:**
```python
class AdaptiveDetector:
    def __init__(self, memory_budget, cpu_budget):
        self.memory_budget = memory_budget
        self.cpu_budget = cpu_budget
        
        # Select detector based on constraints
        if memory_budget < 50:
            self.detector = PSIDetector()  # Memory-efficient
        elif cpu_budget < 0.1:  # seconds
            self.detector = PSIDetector()  # Fast computation
        else:
            self.detector = KSTestDetector()  # Most accurate
    
    def detect(self, data):
        return self.detector.detect(data)
```

**Evaluation:**
- Compare adaptive strategy vs fixed methods
- Show adaptive approach achieves better accuracy-resource trade-off

**Why this is novel:**
- Most research evaluates methods in isolation
- You're proposing a meta-strategy that adapts
- Even simple heuristics can be a contribution if well-evaluated

---

#### **Option 2: Predictive Performance Model**

**Build a model that predicts drift detection performance given constraints.**

**Training data:** Your experimental results

**Input features:**
- Monitoring frequency
- Window size
- Memory budget
- Drift type
- Detector method

**Output:** Predicted detection delay

**Implementation:**
```python
from sklearn.ensemble import GradientBoostingRegressor

# Prepare training data from experiments
X = pd.DataFrame({
    'frequency': [10, 100, 500, ...],
    'window': [50, 100, 500, ...],
    'memory': [10, 50, 100, ...],
    'drift_type': ['sudden', 'gradual', ...],  # one-hot encoded
    'method': ['KS', 'PSI', ...]  # one-hot encoded
})
y = detection_delays

# Train predictor
model = GradientBoostingRegressor()
model.fit(X, y)

# Use for new deployments
new_config = {
    'frequency': 200,
    'window': 300,
    'memory': 75,
    'drift_type': 'gradual',
    'method': 'PSI'
}
predicted_delay = model.predict([new_config])
print(f"Expected detection delay: {predicted_delay} samples")
```

**Why this is valuable:**
- Practitioners can estimate performance before deployment
- Meta-analysis adds a layer of contribution
- Shows you understand the phenomenon deeply enough to model it

---

#### **Option 3: Novel Lightweight Detection Heuristic**

**If you find interesting patterns, propose a simple heuristic.**

**Example scenario:**
> "During experiments, I noticed that for gradual drift under memory constraints, tracking only the moving average and standard deviation of top-3 most important features (by model feature importance) detected drift with 85% accuracy while using 10x less memory than full KS-Test."

**Formalize this as "Importance-Weighted Monitoring (IWM)":**
```python
class ImportanceWeightedDetector:
    def __init__(self, reference_data, model):
        # Get feature importances from model
        importances = model.feature_importances_
        
        # Select top-k features
        self.top_features = np.argsort(importances)[-3:]
        
        # Store lightweight statistics (just mean/std)
        self.ref_stats = {
            feat: {
                'mean': reference_data[feat].mean(),
                'std': reference_data[feat].std()
            }
            for feat in self.top_features
        }
    
    def detect(self, test_data):
        # Compute weighted score
        score = 0
        for feat in self.top_features:
            test_mean = test_data[feat].mean()
            test_std = test_data[feat].std()
            
            # Normalized difference
            mean_diff = abs(test_mean - self.ref_stats[feat]['mean']) / self.ref_stats[feat]['std']
            score += mean_diff
        
        # Threshold
        return score > 1.5, score
```

**Evaluation:**
- Compare IWM to KS-Test and PSI
- Show memory savings vs accuracy trade-off

**Why this could be novel:**
- Leverages model's feature importance (not typical in drift detection)
- Explicit design for resource constraints
- If it works well, it's a small algorithmic contribution

**Caveat:** Only propose this if you empirically observe it's effective. Don't force novelty.

---

### **6.3 Publication-Worthy Contributions Checklist**

**Your project will be publishable if it includes:**

✅ **Empirical Contribution:**
- Comprehensive evaluation across multiple datasets
- Statistical significance testing
- Generalizable findings

✅ **Practical Contribution:**
- Actionable guidelines for practitioners
- Decision frameworks for method selection
- Open-source code and benchmarks

✅ **Novelty (At Least One):**
- [ ] First systematic study of drift detection under resource constraints
- [ ] Novel failure taxonomy and characterization
- [ ] Resource-adaptive detection strategy
- [ ] Predictive performance model
- [ ] New lightweight detection heuristic

✅ **Quality Indicators:**
- Multiple datasets (3+)
- Rigorous experimental design
- Deep analysis explaining "why"
- Clear visualizations
- Reproducible methodology

---

## **7. Month-by-Month Execution Plan**

### **Month 1 (January): Foundation**

**Week 1-2: Literature Review**

**Goal:** Understand the landscape.

**Tasks:**
1. Read 15-20 papers on drift detection
   - Search: "concept drift detection", "data drift monitoring", "covariate shift detection"
   - Focus on: Methods, evaluation metrics, deployment scenarios

2. Read 5-10 papers on resource-constrained ML
   - Search: "edge machine learning", "IoT model deployment", "lightweight ML monitoring"

3. Identify gaps
   - What hasn't been systematically studied?
   - Confirm your project fills a gap

**Deliverable:**
- Literature review chapter draft (10-15 pages)
- Annotated bibliography (notes on each paper)
- List of 2-3 drift detection methods to implement

**Tools:**
- Google Scholar
- Connected Papers (visualize citation networks)
- Notion/Obsidian for note-taking

---

**Week 3: Dataset Selection & Exploration**

**Tasks:**
1. Download 3-4 candidate datasets
2. Exploratory data analysis (EDA)
   - Feature distributions
   - Class balance
   - Missing values
3. Select final datasets based on:
   - Sufficient size (10k+ samples)
   - Diverse domains
   - Clean data (or cleanable)

**Deliverable:**
- Dataset summary report
- EDA notebooks

---

**Week 4: Model Training**

**Tasks:**
1. Train baseline models for each dataset
2. Evaluate baseline performance
3. Save trained models

**Deliverable:**
- Trained models (saved as .pkl files)
- Baseline performance metrics

---

### **Month 2 (February): Implementation**

**Week 1: Drift Simulation**

**Tasks:**
1. Implement drift simulation module
   - Sudden drift function
   - Gradual drift function
   - Recurring drift function
2. Test simulations visually (plot distributions)
3. Validate: Can you "see" the drift in visualizations?

**Deliverable:**
- Working drift simulator
- Visualization of each drift type

---

**Week 2: Drift Detectors**

**Tasks:**
1. Implement KS-Test detector
2. Implement PSI detector
3. (Optional) Implement Wasserstein detector
4. Unit tests: Do they trigger on obvious drift?

**Deliverable:**
- Working detector implementations
- Unit test suite

---

**Week 3-4: Experimental Runner**

**Tasks:**
1. Build experiment configuration system
2. Implement monitoring loop
3. Implement resource tracking
4. Test end-to-end on one configuration

**Deliverable:**
- Fully functional experimental system
- Successfully completed test run

---

### **Month 3 (March): Experimentation Part 1**

**Week 1-2: Baseline Experiments**

**Tasks:**
1. Run unconstrained baseline for all datasets
2. Establish "gold standard" performance
3. Debug any issues

**Deliverable:**
- Baseline results for all datasets
- Confirmed working pipeline

---

**Week 3-4: Constraint Experiments (Frequency)**

**Tasks:**
1. Vary monitoring frequency: {10, 50, 100, 500, 1000}
2. Keep window and memory unlimited
3. Test on all drift types

**Deliverable:**
- Results for frequency ablation study

---

### **Month 4 (April): Experimentation Part 2**

**Week 1: Constraint Experiments (Window Size)**

**Tasks:**
1. Vary window size: {50, 100, 500, 1000, 5000}
2. Keep frequency at baseline, memory unlimited
3. Test on all drift types

**Deliverable:**
- Results for window size ablation study

---

**Week 2: Constraint Experiments (Memory)**

**Tasks:**
1. Vary memory budget: {10, 50, 100, 500 MB}
2. Implement subsampling to enforce limits
3. Test on all drift types

**Deliverable:**
- Results for memory ablation study

---

**Week 3-4: Compound Constraint Experiments**

**Tasks:**
1. Test combinations of constraints
2. Focus on realistic edge-case scenarios
3. Collect final experimental data

**Deliverable:**
- Complete experimental results
- Raw data files (.csv)

---

### **Month 5 (May): Analysis**

**Week 1-2: Data Analysis**

**Tasks:**
1. Statistical analysis (t-tests, regression)
2. Create visualizations (trade-off curves, comparison plots)
3. Identify key patterns and insights

**Deliverable:**
- Analysis scripts
- Figures for thesis

---

**Week 3: Results Chapter**

**Tasks:**
1. Write Results chapter
   - Present findings organized by research question
   - Include tables and figures
   - Report statistical significance

**Deliverable:**
- Results chapter draft (15-20 pages)

---

**Week 4: Analysis Chapter**

**Tasks:**
1. Write Analysis chapter
   - Explain "why" behind patterns
   - Discuss failure modes
   - Derive practitioner guidelines

**Deliverable:**
- Analysis chapter draft (10-15 pages)

---

### **Month 6 (June): Writing Part 1**

**Week 1: Methodology Chapter**

**Tasks:**
1. Document experimental design
2. Describe implementation details
3. Justify design choices

**Deliverable:**
- Methodology chapter (15-20 pages)

---

**Week 2: Introduction & Background**

**Tasks:**
1. Write Introduction (problem statement, motivation)
2. Write Background/Related Work (refined literature review)

**Deliverable:**
- Introduction chapter (5-8 pages)
- Background chapter (revised from Month 1)

---

**Week 3: Conclusion & Future Work**

**Tasks:**
1. Summarize key findings
2. Discuss limitations
3. Propose future research directions

**Deliverable:**
- Conclusion chapter (5-8 pages)

---

**Week 4: Integration & First Draft**

**Tasks:**
1. Combine all chapters
2. Write abstract
3. Create table of contents, references

**Deliverable:**
- Complete first draft (60-80 pages)

---

### **Month 7 (July): Refinement**

**Week 1-2: Revision**

**Tasks:**
1. Read entire thesis cover-to-cover
2. Fix inconsistencies, improve flow
3. Refine figures and tables

**Deliverable:**
- Second draft

---

**Week 3: Supervisor Feedback**

**Tasks:**
1. Submit draft to supervisor
2. Meet for feedback
3. Take detailed notes

**Deliverable:**
- Feedback incorporated list

---

**Week 4: Revisions**

**Tasks:**
1. Implement supervisor feedback
2. Proofread for grammar/spelling
3. Format according to guidelines

**Deliverable:**
- Near-final draft

---

### **Month 8 (August): Finalization**

**Week 1-2: Final Experiments (Optional)**

**Tasks:**
- If supervisor suggests additional experiments
- Or if you want to test the adaptive strategy / novel heuristic

**Deliverable:**
- Supplementary results

---

**Week 3: Final Polish**

**Tasks:**
1. Final proofread
2. Check all citations
3. Format figures professionally

**Deliverable:**
- Final thesis draft

---