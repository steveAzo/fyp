import sys
sys.path.insert(0, '.')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data.loader import load_dataset
from simulators import SuddenDriftSimulator, GradualDriftSimulator, RecurringDriftSimulator
from experiments.grid import PRIMARY_FEATURES

DATASETS = ['credit', 'telco', 'maintenance']
DRIFT_TYPES = ['sudden', 'gradual', 'recurring']
DEPTHS = [3, 5, 10, 20, None]
N_REPS = 5
WINDOW_SIZE = 500          # matches generate_model_comparison_grid
DRIFT_MAGNITUDE = 2.0
DRIFT_START_RATIO = 0.5
BASE_SEED = 42

SIMULATORS = {'sudden': SuddenDriftSimulator, 'gradual': GradualDriftSimulator, 'recurring': RecurringDriftSimulator}

rows = []
for ds in DATASETS:
    data = load_dataset(ds, random_state=BASE_SEED)
    feature = PRIMARY_FEATURES[ds]
    drift_start_idx = max(int(len(data.X_stream) * DRIFT_START_RATIO), WINDOW_SIZE)

    for depth in DEPTHS:
        model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=BASE_SEED, n_jobs=-1)
        model.fit(data.X_reference, data.y_reference)
        baseline_acc = accuracy_score(data.y_stream, model.predict(data.X_stream))

        for drift_type in DRIFT_TYPES:
            for rep in range(N_REPS):
                sim_seed = BASE_SEED + rep
                sim = SIMULATORS[drift_type](random_state=sim_seed)
                X_drifted, _ = sim.apply(data.X_stream, feature=feature, drift_start_idx=drift_start_idx, magnitude=DRIFT_MAGNITUDE)
                post_X = X_drifted.iloc[drift_start_idx:]
                post_y = data.y_stream.iloc[drift_start_idx:]
                post_acc = accuracy_score(post_y, model.predict(post_X)) if len(post_X) > 0 else baseline_acc

                rows.append(dict(
                    dataset=ds, max_depth=str(depth), drift_type=drift_type, repetition=rep,
                    baseline_accuracy=baseline_acc, post_drift_accuracy=post_acc,
                    accuracy_degradation=baseline_acc - post_acc,
                    accuracy_degradation_rel=(baseline_acc - post_acc) / baseline_acc,
                ))

df = pd.DataFrame(rows)
df.to_csv('results/rf_depth_sensitivity.csv', index=False)
print('Saved', len(df), 'rows to results/rf_depth_sensitivity.csv')
print()
print(df.groupby('max_depth')[['accuracy_degradation','accuracy_degradation_rel']].mean().round(4))
