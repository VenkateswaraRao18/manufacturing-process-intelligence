"""
Pre-computes all arrays the Streamlit dashboard needs and saves them
to outputs/data/ as small numpy/json files (~7 MB total).

This allows the dashboard to run on Streamlit Community Cloud
without the raw CSV (151 MB) or model pkl files (85 MB).

Run once locally after training:
    python scripts/precompute.py
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data and models...")
from src.data_loader import DataLoader
import joblib
from sklearn.decomposition import PCA

dl = DataLoader()
dl.load().clean().pivot(min_task_support=0.05)

xgb = joblib.load(ROOT / "outputs" / "models" / "multilabel_xgb.pkl")
km  = joblib.load(ROOT / "outputs" / "models" / "vehicle_kmeans.pkl")

# PCA + cluster labels
print("Computing PCA + cluster labels...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(dl.vehicle_config.astype(np.float32))
cluster_labels = km.predict(X_pca)

# XGBoost predictions (all vehicles)
print("Computing predictions for all vehicles...")
pred_probs = xgb.predict_proba(dl.vehicle_config.astype(np.float32))

# Task counts per vehicle
tasks_per_v = (
    dl.raw_df.groupby("SPNR8")["CHARACTERISTIC"]
    .nunique()
    .reindex(dl.vehicle_ids)
    .values
)

print("Saving arrays...")
np.save(OUT / "vehicle_ids.npy",      dl.vehicle_ids)
np.save(OUT / "vehicle_config.npy",   dl.vehicle_config.astype(np.int8))
np.save(OUT / "vehicle_tasks.npy",    dl.vehicle_tasks.astype(np.int8))
np.save(OUT / "cluster_labels.npy",   cluster_labels.astype(np.int8))
np.save(OUT / "pred_probs.npy",       pred_probs.astype(np.float32))
np.save(OUT / "tasks_per_vehicle.npy", tasks_per_v.astype(np.int16))

with open(OUT / "task_cols.json",   "w") as f:
    json.dump(dl.task_cols, f)
with open(OUT / "config_cols.json", "w") as f:
    json.dump(list(dl.config_cols), f)

# Size report
total = sum((OUT / fn).stat().st_size for fn in [
    "vehicle_ids.npy", "vehicle_config.npy", "vehicle_tasks.npy",
    "cluster_labels.npy", "pred_probs.npy", "tasks_per_vehicle.npy",
    "task_cols.json", "config_cols.json"
])
print(f"\nDone. Total size: {total/1e6:.1f} MB")
for f in sorted(OUT.glob("*")):
    print(f"  {f.name:30s}  {f.stat().st_size/1e3:6.1f} KB")
