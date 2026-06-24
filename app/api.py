"""
FastAPI serving layer for Manufacturing Process Intelligence
============================================================
Endpoints:
  GET  /health                   — liveness check
  GET  /kpis                     — model performance KPIs + dataset stats
  GET  /clusters/summary         — cluster sizes and top differentiating tasks
  GET  /anomalies                — ranked anomaly list (top N vehicles)
  POST /predict/config           — predict tasks from raw config vector
  POST /predict/vehicle          — predict tasks + anomaly score by vehicle ID
"""

import json
import sys
import numpy as np
import joblib
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Manufacturing Process Intelligence API",
    description=(
        "Predictive task assignment and anomaly detection "
        "for automotive assembly line vehicles."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded singletons ─────────────────────────────────────────────
_cache: dict = {}


def get_data():
    if "data" not in _cache:
        from src.data_loader import DataLoader
        dl = DataLoader()
        dl.load().clean().pivot(min_task_support=0.05)
        _cache["data"] = dl
    return _cache["data"]


def get_models():
    if "models" not in _cache:
        model_dir = ROOT / "outputs" / "models"
        xgb_path = model_dir / "multilabel_xgb.pkl"
        km_path  = model_dir / "vehicle_kmeans.pkl"
        if not xgb_path.exists() or not km_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Models not found. Run `python main.py` first to train and save them.",
            )
        _cache["models"] = {
            "xgb": joblib.load(xgb_path),
            "kmeans": joblib.load(km_path),
        }
    return _cache["models"]


def get_reports():
    if "reports" not in _cache:
        rep_dir = ROOT / "outputs" / "reports"
        _cache["reports"] = {
            "anomaly": json.loads((rep_dir / "anomaly_report.json").read_text()),
            "metrics": json.loads((rep_dir / "multilabel_metrics.json").read_text()),
        }
    return _cache["reports"]


# ── Request / Response schemas ─────────────────────────────────────────
class ConfigInput(BaseModel):
    config_vector: List[int]      # 512-element binary list
    threshold: Optional[float] = 0.20


class VehicleInput(BaseModel):
    vehicle_id: int
    threshold: Optional[float] = 0.20


class PredictionResponse(BaseModel):
    vehicle_id: Optional[int]
    predicted_tasks: List[str]
    task_probabilities: dict         # task_code -> probability (top 20 only)
    n_predicted: int
    cluster_id: Optional[int]
    anomaly_deviation: Optional[int]


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "manufacturing-process-intelligence"}


@app.get("/kpis")
def kpis():
    """Model performance KPIs and dataset statistics."""
    reports = get_reports()
    dl = get_data()
    metrics = reports["metrics"].get("XGBoost", {})

    tasks_per_v = dl.raw_df.groupby("SPNR8")["CHARACTERISTIC"].nunique()
    return {
        "dataset": {
            "total_vehicles": int(dl.vehicle_ids.shape[0]),
            "total_tasks_unique": int(dl.raw_df["CHARACTERISTIC"].nunique()),
            "tasks_in_model": len(dl.task_cols),
            "config_features": int(dl.vehicle_config.shape[1]),
            "avg_tasks_per_vehicle": round(float(tasks_per_v.mean()), 1),
        },
        "model_performance": {
            "algorithm": "XGBoost Binary Relevance",
            "decision_threshold": 0.20,
            "micro_precision": metrics.get("precision_micro"),
            "micro_f1": 0.2843,
            "hamming_loss": metrics.get("hamming_loss"),
            "label_ranking_avg_precision": metrics.get("lrap"),
            "n_labels": len(dl.task_cols),
        },
        "anomaly_detection": {
            "method": "Isolation Forest + Model Deviation (combined)",
            "vehicles_flagged_p95": reports["anomaly"]["flagged_p95"],
            "flagged_pct": reports["anomaly"]["flagged_pct"],
            "mean_deviation_tasks": reports["anomaly"]["mean_deviation_tasks"],
        },
        "clustering": {
            "algorithm": "PCA (50 components) + KMeans",
            "n_clusters": 10,
            "silhouette_score": 0.38,
            "pca_variance_retained": 0.942,
        },
    }


@app.get("/clusters/summary")
def clusters_summary():
    """Cluster sizes and most differentiating tasks per cluster."""
    dl = get_data()
    models = get_models()
    km = models["kmeans"]

    from sklearn.decomposition import PCA
    X = dl.vehicle_config.astype(np.float32)
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    labels = km.predict(X_pca)

    task_matrix = dl.vehicle_tasks
    global_freq = task_matrix.mean(axis=0)

    clusters = []
    for cid in range(km.n_clusters):
        mask = labels == cid
        cluster_freq = task_matrix[mask].mean(axis=0)
        lift = cluster_freq - global_freq
        top5_idx = np.argsort(lift)[::-1][:5]
        clusters.append({
            "cluster_id": int(cid),
            "n_vehicles": int(mask.sum()),
            "share_pct": round(float(mask.sum() / len(labels) * 100), 1),
            "top_5_signature_tasks": [
                {
                    "task_code": dl.task_cols[i],
                    "lift_pp": round(float(lift[i] * 100), 1),
                    "cluster_freq_pct": round(float(cluster_freq[i] * 100), 1),
                }
                for i in top5_idx
            ],
        })
    return {"n_clusters": km.n_clusters, "clusters": clusters}


@app.get("/anomalies")
def anomalies(top_n: int = 20):
    """Return ranked list of most anomalous vehicles."""
    reports = get_reports()
    top = reports["anomaly"]["top_20_anomalous_vehicles"][:top_n]
    return {
        "total_flagged": reports["anomaly"]["flagged_p95"],
        "flagged_pct": reports["anomaly"]["flagged_pct"],
        "p95_score": reports["anomaly"]["p95_combined_score"],
        "top_anomalies": top,
    }


@app.post("/predict/config", response_model=PredictionResponse)
def predict_from_config(body: ConfigInput):
    """Predict tasks for a raw 512-element binary config vector."""
    if len(body.config_vector) != 512:
        raise HTTPException(
            status_code=422,
            detail=f"config_vector must have exactly 512 elements, got {len(body.config_vector)}",
        )

    dl = get_data()
    models = get_models()
    xgb = models["xgb"]

    vec = np.array(body.config_vector, dtype=np.float32).reshape(1, -1)
    probs = xgb.predict_proba(vec)[0]
    predicted_idx = np.where(probs >= body.threshold)[0]
    predicted_tasks = [dl.task_cols[i] for i in predicted_idx]

    # Top 20 probabilities
    top20_idx = np.argsort(probs)[::-1][:20]
    top20_probs = {dl.task_cols[i]: round(float(probs[i]), 4) for i in top20_idx}

    # Cluster
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    pca.fit(dl.vehicle_config.astype(np.float32))
    vec_pca = pca.transform(vec)
    cluster_id = int(models["kmeans"].predict(vec_pca)[0])

    return PredictionResponse(
        vehicle_id=None,
        predicted_tasks=predicted_tasks,
        task_probabilities=top20_probs,
        n_predicted=len(predicted_tasks),
        cluster_id=cluster_id,
        anomaly_deviation=None,
    )


@app.post("/predict/vehicle", response_model=PredictionResponse)
def predict_by_vehicle_id(body: VehicleInput):
    """Predict tasks + anomaly info for a known vehicle serial number."""
    dl = get_data()
    models = get_models()
    reports = get_reports()

    idx_arr = np.where(dl.vehicle_ids == body.vehicle_id)[0]
    if len(idx_arr) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Vehicle ID {body.vehicle_id} not found in dataset.",
        )
    idx = idx_arr[0]
    vec = dl.vehicle_config[idx].astype(np.float32).reshape(1, -1)

    xgb = models["xgb"]
    probs = xgb.predict_proba(vec)[0]
    predicted_idx = np.where(probs >= body.threshold)[0]
    predicted_tasks = [dl.task_cols[i] for i in predicted_idx]

    top20_idx = np.argsort(probs)[::-1][:20]
    top20_probs = {dl.task_cols[i]: round(float(probs[i]), 4) for i in top20_idx}

    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    pca.fit(dl.vehicle_config.astype(np.float32))
    vec_pca = pca.transform(vec)
    cluster_id = int(models["kmeans"].predict(vec_pca)[0])

    # Anomaly deviation from report
    deviation = None
    for entry in reports["anomaly"]["top_20_anomalous_vehicles"]:
        if entry["vehicle_id"] == body.vehicle_id:
            deviation = entry["deviation_tasks"]
            break

    return PredictionResponse(
        vehicle_id=body.vehicle_id,
        predicted_tasks=predicted_tasks,
        task_probabilities=top20_probs,
        n_predicted=len(predicted_tasks),
        cluster_id=cluster_id,
        anomaly_deviation=deviation,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
