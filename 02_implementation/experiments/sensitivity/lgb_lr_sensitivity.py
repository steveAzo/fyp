import sys
sys.path.insert(0, '.')
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from data.loader import load_dataset
from simulators import SuddenDriftSimulator, GradualDriftSimulator, RecurringDriftSimulator
from experiments.grid import PRIMARY_FEATURES
import re

DATASETS = ['credit', 'telco', 'maintenance']
DRIFT_TYPES = ['sudden', 'gradual', 'recurring']
LEARNING_RATES = [0.05, 0.1, 0.2]
N_REPS = 5
WINDOW_SIZE = 500
DRIFT_MAGNITUDE = 2.0
DRIFT_START_RATIO = 0.5
BASE_SEED = 42

SIMULATORS = {'sudden': SuddenDriftSimulator, 'gradual': GradualDriftSimulator, 'recurring': RecurringDriftSimulator}

def safe_cols(df):
    mapping = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns}
    return df.rename(columns=mapping), mapping

rows = []
for ds in DATASETS:
    data = load_dataset(ds, random_state=BASE_SEED)
    feature = PRIMARY_FEATURES[ds]
    drift_start_idx = max(int(len(data.X_stream) * DRIFT_START_RATIO), WINDOW_SIZE)
    Xr, mapping = safe_cols(data.X_reference)
    Xs = data.X_stream.rename(columns=mapping)

    for lr in LEARNING_RATES:
        model = LGBMClassifier(n_estimators=100, learning_rate=lr, random_state=BASE_SEED, n_jobs=-1, verbose=-1)
        model.fit(Xr, data.y_reference)
        baseline_acc = accuracy_score(data.y_stream, model.predict(Xs))

        for drift_type in DRIFT_TYPES:
            for rep in range(N_REPS):
                sim_seed = BASE_SEED + rep
                sim = SIMULATORS[drift_type](random_state=sim_seed)
                X_drifted, _ = sim.apply(data.X_stream, feature=feature, drift_start_idx=drift_start_idx, magnitude=DRIFT_MAGNITUDE)
                X_drifted = X_drifted.rename(columns=mapping)
                post_X = X_drifted.iloc[drift_start_idx:]
                post_y = data.y_stream.iloc[drift_start_idx:]
                post_acc = accuracy_score(post_y, model.predict(post_X)) if len(post_X) > 0 else baseline_acc

                rows.append(dict(
                    dataset=ds, learning_rate=lr, drift_type=drift_type, repetition=rep,
                    accuracy_degradation=baseline_acc - post_acc,
                ))

df = pd.DataFrame(rows)
df.to_csv('results/lgb_lr_sensitivity.csv', index=False)
print('Saved', len(df), 'rows')
print(df.groupby('learning_rate')['accuracy_degradation'].mean().round(4))
