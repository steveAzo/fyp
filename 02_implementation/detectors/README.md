# Detectors Module

## Purpose
Implement drift detection methods: KS-Test, PSI, and Wasserstein distance.

## Usage

```python
from detectors.ks_test import KSTestDetector
from detectors.psi import PSIDetector

# Create detector
ks_detector = KSTestDetector(threshold=0.05)

# Detect drift
is_drift, p_value = ks_detector.detect(
    reference_data=X_train['age'].values,
    test_data=X_test['age'].values
)

if is_drift:
    print(f"Drift detected! p-value: {p_value}")
```

## Available Detectors

### 1. Kolmogorov-Smirnov Test (`ks_test.py`)
**File**: `ks_test.py`

**Parameters**:
- `threshold` (float): p-value threshold (default: 0.05)

**Method**: Compares cumulative distributions, returns p-value

**Best for**: Clean data, univariate features, covariate drift

**Computational cost**: O(n log n)

**Memory cost**: O(n) - stores full reference distribution

**Pros**:
- Non-parametric (no distribution assumptions)
- Interpretable p-values
- Well-established statistical test

**Cons**:
- Univariate only (test each feature separately)
- Requires large samples for stability
- Sensitive to sample size

### 2. Population Stability Index (`psi.py`)
**File**: `psi.py`

**Parameters**:
- `threshold` (float): PSI score threshold (default: 0.1)
- `n_bins` (int): Number of bins for histogram (default: 10)

**Method**: Bins distribution, measures proportion shift

**Best for**: Resource-constrained scenarios, proven in practice (finance)

**Computational cost**: O(n)

**Memory cost**: O(k) where k = number of bins (~10-20)

**Pros**:
- Single interpretable score
- Very low memory requirement
- Widely used in industry (finance, banking)
- Fast computation

**Cons**:
- Requires binning decision
- Less sensitive to distribution shape
- May miss within-bin drift

**PSI Interpretation**:
- < 0.1: No significant shift
- 0.1-0.25: Moderate shift, investigate
- > 0.25: Significant shift, likely drift

### 3. Wasserstein Distance (`wasserstein.py` - To Implement)
**Method**: Optimal transport distance between distributions

**Best for**: Complex distributions, when shape matters

**Computational cost**: O(n^3) standard, O(n log n) with approximations

**Memory cost**: O(n)

**Pros**:
- Captures distribution geometry
- Effective for gradual drift
- Continuous distance measure

**Cons**:
- Computationally expensive
- Less interpretable than PSI
- Overkill for simple drift patterns

## Common Interface

All detectors inherit from `DriftDetector` base class:

```python
class DriftDetector(ABC):
    def __init__(self, threshold):
        self.threshold = threshold
    
    @abstractmethod
    def detect(self, reference_data, test_data, feature=None):
        """Returns: (is_drift: bool, score: float)"""
        pass
```

This ensures consistent API across all detectors.

## Implementation Notes

1. **Handle pandas Series**: Detectors accept both numpy arrays and pandas Series
2. **Remove NaN values**: Detectors automatically remove NaNs
3. **Univariate**: Test each feature separately (for feature drift analysis)
4. **Logging**: Detectors can log scores for visualization

## Testing Detectors

```python
# Test on synthetic data with known drift
import numpy as np

reference = np.random.normal(loc=0, scale=1, size=1000)
no_drift = np.random.normal(loc=0, scale=1, size=500)  # Same distribution
drift = np.random.normal(loc=2, scale=1, size=500)     # Different mean

detector = KSTestDetector(threshold=0.05)

is_drift_no, score_no = detector.detect(reference, no_drift)
is_drift_yes, score_yes = detector.detect(reference, drift)

assert not is_drift_no  # Should not detect drift
assert is_drift_yes      # Should detect drift
print("✓ Detector tests passed!")
```

## Future Extensions

1. **Multivariate detectors**: Test multiple features simultaneously
2. **Adaptive thresholds**: Learn thresholds from data
3. **Feature importance**: Weight detections by feature importance
4. **Concept drift detectors**: For non-stationary label distributions

