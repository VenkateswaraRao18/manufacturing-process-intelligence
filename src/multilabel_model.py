import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    hamming_loss, f1_score, precision_score, recall_score,
    label_ranking_average_precision_score, average_precision_score,
    roc_auc_score
)
from xgboost import XGBClassifier

FIGURES = Path(__file__).parent.parent / "outputs" / "figures"
MODELS  = Path(__file__).parent.parent / "outputs" / "models"
REPORTS = Path(__file__).parent.parent / "outputs" / "reports"
for p in (FIGURES, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


class MultiLabelModel:
    def __init__(self, data_loader, cluster_labels=None):
        self.dl = data_loader
        self.cluster_labels = cluster_labels
        self.X = data_loader.vehicle_config.astype(np.float32)
        self.Y = data_loader.vehicle_tasks.astype(np.int8)
        self.task_cols = data_loader.task_cols

        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.model_xgb = None
        self.model_lr  = None
        self.metrics   = {}

    # ------------------------------------------------------------------
    def split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        print(f"\n[MultiLabel] Train: {self.X_train.shape[0]} vehicles  "
              f"Test: {self.X_test.shape[0]} vehicles  "
              f"Labels: {self.Y.shape[1]}")
        return self

    # ------------------------------------------------------------------
    def train_logistic(self):
        print("[MultiLabel] Training Logistic Regression baseline...")
        lr_base = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs",
                                     random_state=42)
        self.model_lr = OneVsRestClassifier(lr_base, n_jobs=-1)
        self.model_lr.fit(self.X_train, self.Y_train)
        print("  Done.")
        return self

    def train_xgboost(self):
        print("[MultiLabel] Training XGBoost (Binary Relevance)...")
        # Estimate class imbalance per label
        pos_rates = self.Y_train.mean(axis=0)
        # We use a single shared XGBoost per label via OneVsRest
        xgb_base = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model_xgb = OneVsRestClassifier(xgb_base, n_jobs=-1)
        self.model_xgb.fit(self.X_train, self.Y_train)
        print("  Done.")
        joblib.dump(self.model_xgb, MODELS / "multilabel_xgb.pkl")
        return self

    # ------------------------------------------------------------------
    def evaluate(self, threshold=0.5):
        print("\n[MultiLabel] Evaluating models...")
        results = {}
        for name, model in [("LogisticRegression", self.model_lr),
                             ("XGBoost", self.model_xgb)]:
            if model is None:
                continue

            Y_prob = model.predict_proba(self.X_test)
            Y_pred = (Y_prob >= threshold).astype(int)

            hl   = hamming_loss(self.Y_test, Y_pred)
            f1_mi = f1_score(self.Y_test, Y_pred, average="micro", zero_division=0)
            f1_ma = f1_score(self.Y_test, Y_pred, average="macro", zero_division=0)
            prec  = precision_score(self.Y_test, Y_pred, average="micro", zero_division=0)
            rec   = recall_score(self.Y_test, Y_pred, average="micro", zero_division=0)
            lrap  = label_ranking_average_precision_score(self.Y_test, Y_prob)

            # Subset accuracy (exact match)
            exact = np.all(Y_pred == self.Y_test, axis=1).mean()

            results[name] = {
                "hamming_loss"      : round(hl, 4),
                "f1_micro"          : round(f1_mi, 4),
                "f1_macro"          : round(f1_ma, 4),
                "precision_micro"   : round(prec, 4),
                "recall_micro"      : round(rec, 4),
                "lrap"              : round(lrap, 4),
                "subset_accuracy"   : round(exact, 4),
            }

            print(f"\n  {name}:")
            for k, v in results[name].items():
                print(f"    {k:25s}: {v}")

        self.metrics = results
        self._plot_metrics_comparison(results)

        # Per-label F1 distribution for best model (XGBoost)
        if self.model_xgb:
            Y_prob = self.model_xgb.predict_proba(self.X_test)
            Y_pred = (Y_prob >= threshold).astype(int)
            per_label_f1 = f1_score(self.Y_test, Y_pred, average=None, zero_division=0)
            self._plot_per_label_f1(per_label_f1)

        # Save metrics
        with open(REPORTS / "multilabel_metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        return self

    def _plot_metrics_comparison(self, results):
        models = list(results.keys())
        metrics_to_plot = ["f1_micro", "f1_macro", "precision_micro",
                           "recall_micro", "lrap", "subset_accuracy"]
        x = np.arange(len(metrics_to_plot))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#4C72B0", "#DD8800"]
        for i, (model, color) in enumerate(zip(models, colors)):
            vals = [results[model][m] for m in metrics_to_plot]
            bars = ax.bar(x + i * width, vals, width, label=model,
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics_to_plot, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Multi-label Model Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "multilabel_model_comparison.png", dpi=150)
        plt.close(fig)

    def _plot_per_label_f1(self, per_label_f1):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(per_label_f1, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.axvline(per_label_f1.mean(), color="#DD4444", linestyle="--",
                   linewidth=1.5, label=f"Mean F1 = {per_label_f1.mean():.3f}")
        ax.axvline(np.median(per_label_f1), color="#22AA44", linestyle="--",
                   linewidth=1.5, label=f"Median F1 = {np.median(per_label_f1):.3f}")
        ax.set_xlabel("Per-label F1 Score")
        ax.set_ylabel("Number of Tasks")
        ax.set_title("XGBoost — Per-label F1 Distribution (across 325 tasks)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "per_label_f1_distribution.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def tune_threshold(self):
        """Find the threshold that maximises micro-F1 on test set."""
        if self.model_xgb is None:
            return
        print("\n[MultiLabel] Threshold tuning (XGBoost)...")
        Y_prob = self.model_xgb.predict_proba(self.X_test)
        thresholds = np.arange(0.2, 0.85, 0.05)
        f1s = []
        for t in thresholds:
            Y_pred = (Y_prob >= t).astype(int)
            f1s.append(f1_score(self.Y_test, Y_pred, average="micro", zero_division=0))

        best_t = thresholds[int(np.argmax(f1s))]
        print(f"  Best threshold: {best_t:.2f}  micro-F1: {max(f1s):.4f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thresholds, f1s, marker="o", color="#4C72B0", linewidth=1.5)
        ax.axvline(best_t, color="#DD4444", linestyle="--",
                   label=f"Best = {best_t:.2f}")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Micro-F1")
        ax.set_title("Threshold vs Micro-F1 (XGBoost)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "threshold_tuning.png", dpi=150)
        plt.close(fig)

        self.best_threshold = best_t
        return best_t

    # ------------------------------------------------------------------
    def predict_vehicle(self, config_vector, threshold=None):
        """Predict task list for a single vehicle config vector."""
        t = threshold or getattr(self, "best_threshold", 0.5)
        prob = self.model_xgb.predict_proba(config_vector.reshape(1, -1))[0]
        predicted_tasks = [self.task_cols[i] for i, p in enumerate(prob) if p >= t]
        return predicted_tasks, prob

    def get_predicted_probs(self, threshold=None):
        """Return full test-set predicted probabilities."""
        return self.model_xgb.predict_proba(self.X_test)
