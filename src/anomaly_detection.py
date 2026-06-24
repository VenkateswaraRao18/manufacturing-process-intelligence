import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FIGURES = Path(__file__).parent.parent / "outputs" / "figures"
REPORTS = Path(__file__).parent.parent / "outputs" / "reports"
for p in (FIGURES, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


class AnomalyDetector:
    def __init__(self, data_loader, ml_model, cluster_labels=None):
        self.dl = data_loader
        self.ml = ml_model
        self.cluster_labels = cluster_labels

        self.X = data_loader.vehicle_config.astype(np.float32)
        self.Y = data_loader.vehicle_tasks.astype(np.int8)
        self.task_cols = data_loader.task_cols
        self.vehicle_ids = data_loader.vehicle_ids

        self.iso_scores = None        # isolation forest anomaly scores (all vehicles)
        self.deviation_scores = None  # model-based task deviation scores
        self.combined_scores = None

    # ------------------------------------------------------------------
    # 1. Isolation Forest on config features
    # ------------------------------------------------------------------
    def fit_isolation_forest(self, contamination=0.05):
        print("\n[Anomaly] Fitting Isolation Forest on config features...")
        iso = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        # Use PCA-reduced features for stability
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(self.X)

        iso.fit(X_pca)
        # score_samples: higher (less negative) = more normal
        raw_scores = iso.score_samples(X_pca)
        # Invert so higher = more anomalous
        self.iso_scores = -raw_scores
        self.iso_predictions = iso.predict(X_pca)   # -1=anomaly, 1=normal

        n_anomalies = (self.iso_predictions == -1).sum()
        print(f"  Flagged {n_anomalies} vehicles as anomalies "
              f"({n_anomalies/len(self.vehicle_ids)*100:.1f}%)")
        return self

    # ------------------------------------------------------------------
    # 2. Model-deviation scoring (all vehicles via cross-val-like approach)
    # ------------------------------------------------------------------
    def compute_deviation_scores(self, threshold=0.20):
        """
        For each vehicle, compute deviation = number of tasks where the model's
        prediction (thresholded) disagrees with the actual task assignment.
        Vehicles with high deviation performed tasks unexpected for their config,
        or skipped tasks that were predicted — both are anomalous.
        """
        print("[Anomaly] Computing model-deviation scores on full dataset...")

        if self.ml.model_xgb is None:
            raise RuntimeError("Train XGBoost model first.")

        Y_prob_all = self.ml.model_xgb.predict_proba(self.X)
        Y_pred_all = (Y_prob_all >= threshold).astype(int)

        # Per-vehicle disagreement count
        disagreement = np.abs(Y_pred_all - self.Y).sum(axis=1)

        # Decompose into: unexpected tasks & missed tasks
        unexpected = ((Y_pred_all == 0) & (self.Y == 1)).sum(axis=1)  # model said no, vehicle did it
        missed     = ((Y_pred_all == 1) & (self.Y == 0)).sum(axis=1)  # model said yes, vehicle skipped it

        self.deviation_scores = disagreement.astype(float)
        self.unexpected_counts = unexpected
        self.missed_counts = missed

        print(f"  Mean deviation   : {disagreement.mean():.1f} tasks")
        print(f"  Median deviation : {np.median(disagreement):.1f} tasks")
        print(f"  Max deviation    : {disagreement.max()} tasks")
        return self

    # ------------------------------------------------------------------
    # 3. Combined score + ranking
    # ------------------------------------------------------------------
    def combine_scores(self):
        print("[Anomaly] Computing combined anomaly scores...")

        # Normalise each score to [0, 1]
        def norm(x):
            r = x - x.min()
            return r / (r.max() + 1e-9)

        iso_n = norm(self.iso_scores)
        dev_n = norm(self.deviation_scores)

        # Weighted average: give more weight to model deviation (richer signal)
        self.combined_scores = 0.35 * iso_n + 0.65 * dev_n

        # Rank vehicles
        ranked_idx = np.argsort(self.combined_scores)[::-1]
        self.ranked_vehicle_ids = self.vehicle_ids[ranked_idx]
        self.ranked_scores = self.combined_scores[ranked_idx]

        top10_idx = ranked_idx[:10]
        print("\n  Top 10 most anomalous vehicles:")
        print(f"  {'VehicleID':>12}  {'Combined':>10}  {'Deviation':>10}  "
              f"{'IsoForest':>10}  {'Unexpected':>12}  {'Missed':>8}")
        for i in top10_idx:
            print(f"  {self.vehicle_ids[i]:>12}  "
                  f"{self.combined_scores[i]:>10.4f}  "
                  f"{self.deviation_scores[i]:>10.0f}  "
                  f"{self.iso_scores[i]:>10.4f}  "
                  f"{self.unexpected_counts[i]:>12}  "
                  f"{self.missed_counts[i]:>8}")
        return self

    # ------------------------------------------------------------------
    # 4. Visualisations
    # ------------------------------------------------------------------
    def plot_all(self):
        self._plot_score_distributions()
        self._plot_combined_vs_deviation()
        self._plot_anomaly_umap()
        self._plot_top_anomalous_tasks()
        print(f"\n[Anomaly] All plots saved to {FIGURES}")

    def _plot_score_distributions(self):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for ax, scores, title, color in [
            (axes[0], self.iso_scores,         "Isolation Forest Score", "#4C72B0"),
            (axes[1], self.deviation_scores,   "Model Deviation Score",  "#DD8800"),
            (axes[2], self.combined_scores,    "Combined Anomaly Score", "#AA4488"),
        ]:
            q95 = np.percentile(scores, 95)
            ax.hist(scores, bins=40, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
            ax.axvline(q95, color="#CC2222", linestyle="--", linewidth=1.3,
                       label=f"P95 = {q95:.2f}")
            ax.set_title(title)
            ax.set_xlabel("Score")
            ax.set_ylabel("Vehicles")
            ax.legend(fontsize=8)

        fig.suptitle("Anomaly Score Distributions", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES / "anomaly_score_distributions.png", dpi=150)
        plt.close(fig)

    def _plot_combined_vs_deviation(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        is_anomaly = (self.iso_predictions == -1)

        ax.scatter(self.deviation_scores[~is_anomaly],
                   self.combined_scores[~is_anomaly],
                   s=15, alpha=0.4, color="#4C72B0", label="Normal")
        ax.scatter(self.deviation_scores[is_anomaly],
                   self.combined_scores[is_anomaly],
                   s=25, alpha=0.8, color="#DD3333", label="Isolation Forest Anomaly",
                   marker="^")

        # Annotate top anomalies
        top5_idx = np.argsort(self.combined_scores)[::-1][:5]
        for i in top5_idx:
            ax.annotate(str(self.vehicle_ids[i]),
                        (self.deviation_scores[i], self.combined_scores[i]),
                        fontsize=6.5, xytext=(4, 2), textcoords="offset points")

        ax.set_xlabel("Model Deviation Score (tasks)")
        ax.set_ylabel("Combined Anomaly Score")
        ax.set_title("Anomaly Detection: Deviation vs Combined Score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "anomaly_deviation_vs_combined.png", dpi=150)
        plt.close(fig)

    def _plot_anomaly_umap(self):
        # Only available when clustering ran
        try:
            from src.clustering import VehicleClusterer
        except ImportError:
            return

        if not hasattr(self.ml, 'umap_coords') or self.ml.umap_coords is None:
            return  # skip if UMAP not run

    def _plot_top_anomalous_tasks(self):
        """Which tasks appear most in anomalous vehicles (unexpected tasks)?"""
        top_anomaly_idx = np.argsort(self.combined_scores)[::-1][:100]
        normal_idx = np.argsort(self.combined_scores)[:100]   # bottom 100 = most normal

        anomalous_task_freq = self.Y[top_anomaly_idx].mean(axis=0) * 100
        normal_task_freq    = self.Y[normal_idx].mean(axis=0) * 100
        lift = anomalous_task_freq - normal_task_freq

        top10_idx = np.argsort(lift)[::-1][:10]
        bot10_idx = np.argsort(lift)[:10]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.barh([self.task_cols[i][:22] for i in top10_idx[::-1]],
                lift[top10_idx[::-1]], color="#DD4444", alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Tasks Over-represented in\nTop 100 Anomalous Vehicles")
        ax.set_xlabel("Frequency Lift (pp vs normal vehicles)")
        ax.tick_params(axis="y", labelsize=7)

        ax2 = axes[1]
        ax2.barh([self.task_cols[i][:22] for i in bot10_idx[::-1]],
                 lift[bot10_idx[::-1]], color="#4C72B0", alpha=0.8)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_title("Tasks Under-represented in\nTop 100 Anomalous Vehicles")
        ax2.set_xlabel("Frequency Lift (pp vs normal vehicles)")
        ax2.tick_params(axis="y", labelsize=7)

        fig.tight_layout()
        fig.savefig(FIGURES / "anomaly_task_signatures.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Save report
    # ------------------------------------------------------------------
    def save_report(self):
        top_idx = np.argsort(self.combined_scores)[::-1]

        # P95 threshold for flagging
        p95 = float(np.percentile(self.combined_scores, 95))
        flagged = (self.combined_scores >= p95).sum()

        report = {
            "total_vehicles": int(len(self.vehicle_ids)),
            "flagged_p95": int(flagged),
            "flagged_pct": round(float(flagged / len(self.vehicle_ids) * 100), 2),
            "mean_deviation_tasks": round(float(self.deviation_scores.mean()), 2),
            "p95_combined_score": round(p95, 4),
            "top_20_anomalous_vehicles": [
                {
                    "vehicle_id": int(self.vehicle_ids[i]),
                    "combined_score": round(float(self.combined_scores[i]), 4),
                    "deviation_tasks": int(self.deviation_scores[i]),
                    "unexpected_tasks": int(self.unexpected_counts[i]),
                    "missed_tasks": int(self.missed_counts[i]),
                }
                for i in top_idx[:20]
            ],
        }

        with open(REPORTS / "anomaly_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n[Anomaly] Report saved → outputs/reports/anomaly_report.json")
        print(f"  Flagged (P95): {flagged} vehicles ({report['flagged_pct']}%)")
        return report
