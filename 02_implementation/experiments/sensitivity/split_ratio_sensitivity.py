import sys
sys.path.insert(0, '.')
import pandas as pd
from data.loader import load_dataset
from detectors import KSTestDetector, PSIDetector, WassersteinDetector
from simulators import SuddenDriftSimulator, GradualDriftSimulator, RecurringDriftSimulator
from experiments.grid import PRIMARY_FEATURES

DATASETS = ['credit', 'telco', 'maintenance']
DETECTORS = {'ks': lambda: KSTestDetector(threshold=0.05),
             'psi': lambda: PSIDetector(threshold=0.25, n_bins=10),
             'wasserstein': lambda: WassersteinDetector(threshold=0.25)}
DRIFT_TYPES = ['sudden', 'gradual', 'recurring']
SPLITS = [0.5, 0.6, 0.7]
N_REPS = 5
MONITORING_FREQ = 100
WINDOW_SIZE = 500
DRIFT_MAGNITUDE = 2.0
DRIFT_START_RATIO = 0.5
BASE_SEED = 42

SIMULATORS = {'sudden': SuddenDriftSimulator, 'gradual': GradualDriftSimulator, 'recurring': RecurringDriftSimulator}

rows = []
for split in SPLITS:
    for ds in DATASETS:
        data = load_dataset(ds, reference_ratio=split, random_state=BASE_SEED)
        feature = PRIMARY_FEATURES[ds]
        drift_start_idx = max(int(len(data.X_stream) * DRIFT_START_RATIO), WINDOW_SIZE)
        ref_array = data.X_reference[feature].values

        for det_name, det_factory in DETECTORS.items():
            for drift_type in DRIFT_TYPES:
                for rep in range(N_REPS):
                    sim_seed = BASE_SEED + rep
                    sim = SIMULATORS[drift_type](random_state=sim_seed)
                    X_drifted, _ = sim.apply(data.X_stream, feature=feature, drift_start_idx=drift_start_idx, magnitude=DRIFT_MAGNITUDE)
                    stream_arr = X_drifted[feature].values
                    n_samples = len(stream_arr)

                    detector = det_factory()
                    drift_detected_at = None
                    for check_idx in range(WINDOW_SIZE, n_samples, MONITORING_FREQ):
                        test_window = stream_arr[check_idx - WINDOW_SIZE: check_idx]
                        result = detector.detect(ref_array, test_window, feature=feature)
                        if result.is_drift and check_idx >= drift_start_idx:
                            drift_detected_at = check_idx
                            break

                    delay = drift_detected_at - drift_start_idx if drift_detected_at is not None else None
                    rows.append(dict(split_ratio=split, dataset=ds, detector=det_name, drift_type=drift_type,
                                      repetition=rep, detection_delay=delay, missed=delay is None))

df = pd.DataFrame(rows)
df.to_csv('results/split_ratio_sensitivity.csv', index=False)
print('Saved', len(df), 'rows')
print()
det = df[~df.missed]
print(det.groupby(['split_ratio','detector'])['detection_delay'].mean().round(1))
print()
print('Missed detection rate by split:')
print(df.groupby('split_ratio')['missed'].mean().round(4))
