import sys
sys.path.insert(0, '.')
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from data.loader import load_dataset
from simulators import SuddenDriftSimulator, GradualDriftSimulator, RecurringDriftSimulator
from experiments.grid import PRIMARY_FEATURES

DATASETS = ['credit', 'telco', 'maintenance']
DRIFT_TYPES = ['sudden', 'gradual', 'recurring']
MODELS = ['random_forest', 'logistic_regression', 'lightgbm']
N_REPS = 5
WINDOW_SIZE = 500
DRIFT_MAGNITUDE = 2.0
DRIFT_START_RATIO = 0.5
BASE_SEED = 42

SIMULATORS = {'sudden': SuddenDriftSimulator, 'gradual': GradualDriftSimulator, 'recurring': RecurringDriftSimulator}

def safe_cols(df):
    mapping = {c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns}
    return df.rename(columns=mapping), mapping

def make_model(model_type, seed):
    if model_type == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if model_type == 'logistic_regression':
        return LogisticRegression(max_iter=1000, random_state=seed)
    if model_type == 'lightgbm':
        return LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(model_type)

def eval_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return acc, bal, auc

rows = []
for ds in DATASETS:
    data = load_dataset(ds, random_state=BASE_SEED)
    feature = PRIMARY_FEATURES[ds]
    drift_start_idx = max(int(len(data.X_stream) * DRIFT_START_RATIO), WINDOW_SIZE)

    for model_type in MODELS:
        model = make_model(model_type, BASE_SEED)
        if model_type == 'lightgbm':
            Xr, mapping = safe_cols(data.X_reference)
            Xs = data.X_stream.rename(columns=mapping)
        else:
            Xr, Xs, mapping = data.X_reference, data.X_stream, None

        model.fit(Xr, data.y_reference)

        base_pred = model.predict(Xs)
        base_prob = model.predict_proba(Xs)[:, 1]
        base_acc, base_bal, base_auc = eval_metrics(data.y_stream, base_pred, base_prob)

        for drift_type in DRIFT_TYPES:
            for rep in range(N_REPS):
                sim_seed = BASE_SEED + rep
                sim = SIMULATORS[drift_type](random_state=sim_seed)
                X_drifted, _ = sim.apply(data.X_stream, feature=feature, drift_start_idx=drift_start_idx, magnitude=DRIFT_MAGNITUDE)
                if mapping:
                    X_drifted = X_drifted.rename(columns=mapping)
                post_X = X_drifted.iloc[drift_start_idx:]
                post_y = data.y_stream.iloc[drift_start_idx:]

                if len(post_X) > 0:
                    post_pred = model.predict(post_X)
                    post_prob = model.predict_proba(post_X)[:, 1]
                    post_acc, post_bal, post_auc = eval_metrics(post_y, post_pred, post_prob)
                else:
                    post_acc, post_bal, post_auc = base_acc, base_bal, base_auc

                rows.append(dict(
                    dataset=ds, model=model_type, drift_type=drift_type, repetition=rep,
                    baseline_accuracy=base_acc, post_drift_accuracy=post_acc, degradation_acc=base_acc-post_acc,
                    baseline_balanced_acc=base_bal, post_drift_balanced_acc=post_bal, degradation_balanced_acc=base_bal-post_bal,
                    baseline_auc=base_auc, post_drift_auc=post_auc, degradation_auc=base_auc-post_auc,
                ))

df = pd.DataFrame(rows)
df.to_csv('results/model_comparison_extra_metrics.csv', index=False)
print('Saved', len(df), 'rows')
print()
print(df.groupby('model')[['degradation_acc','degradation_balanced_acc','degradation_auc']].mean().round(4))
