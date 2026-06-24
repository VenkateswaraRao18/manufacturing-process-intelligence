"""
Manufacturing Process Intelligence
===================================
End-to-end runner: Data → EDA → Clustering → Multi-label Model → Anomaly Detection
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.eda import EDAAnalyzer
from src.clustering import VehicleClusterer
from src.multilabel_model import MultiLabelModel
from src.anomaly_detection import AnomalyDetector


def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    t0 = time.time()

    # ── 1. Data Loading ──────────────────────────────────────────
    banner("STEP 1 — Data Loading & Preprocessing")
    dl = DataLoader()
    dl.load().clean().pivot(min_task_support=0.05)
    dl.summarize()

    # ── 2. EDA ───────────────────────────────────────────────────
    banner("STEP 2 — Exploratory Data Analysis")
    eda = EDAAnalyzer(dl)
    eda.run_all()

    # ── 3. Clustering ─────────────────────────────────────────────
    banner("STEP 3 — Vehicle Clustering (PCA + K-Means + UMAP)")
    vc = VehicleClusterer(dl)
    vc.run_pca(n_components=50)
    vc.fit()             # finds optimal K automatically
    vc.run_umap()
    vc.plot_cluster_task_profiles()
    cluster_labels = vc.get_cluster_labels()

    # ── 4. Multi-label Model ──────────────────────────────────────
    banner("STEP 4 — Multi-label Task Prediction (XGBoost)")
    ml = MultiLabelModel(dl, cluster_labels=cluster_labels)
    ml.split()
    ml.train_logistic()
    ml.train_xgboost()
    ml.evaluate()
    best_t = ml.tune_threshold()

    # ── 5. Anomaly Detection ──────────────────────────────────────
    banner("STEP 5 — Anomaly Detection")
    ad = AnomalyDetector(dl, ml, cluster_labels=cluster_labels)
    ad.fit_isolation_forest(contamination=0.05)
    ad.compute_deviation_scores(threshold=best_t)
    ad.combine_scores()
    ad.plot_all()
    report = ad.save_report()

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner("COMPLETE")
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print(f"\nOutputs:")
    print(f"  Figures → outputs/figures/   ({len(list(Path('outputs/figures').glob('*.png')))} PNGs)")
    print(f"  Models  → outputs/models/")
    print(f"  Reports → outputs/reports/")
    print(f"\nKey Results:")
    metrics = ml.metrics.get("XGBoost", {})
    print(f"  XGBoost micro-F1 (threshold={best_t:.2f}): "
          f"{metrics.get('f1_micro','?')}")
    print(f"  XGBoost precision: {metrics.get('precision_micro','?')}")
    print(f"  Anomalous vehicles (P95): {report['flagged_p95']} "
          f"({report['flagged_pct']}%)")
    print(f"  Vehicle clusters found: {vc.k}")


if __name__ == "__main__":
    main()
