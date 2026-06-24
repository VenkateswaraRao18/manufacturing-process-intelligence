import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import umap

FIGURES = Path(__file__).parent.parent / "outputs" / "figures"
MODELS  = Path(__file__).parent.parent / "outputs" / "models"
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
PALETTE = sns.color_palette("tab10")


class VehicleClusterer:
    def __init__(self, data_loader):
        self.dl = data_loader
        self.X = data_loader.vehicle_config.astype(np.float32)
        self.pca_coords = None
        self.umap_coords = None
        self.labels = None
        self.k = None
        self.kmeans = None

    # ------------------------------------------------------------------
    def run_pca(self, n_components=50):
        print("\n[Clustering] Running PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        self.pca_coords = pca.fit_transform(self.X)
        explained = pca.explained_variance_ratio_.cumsum()
        print(f"  PCA {n_components} components explain "
              f"{explained[-1]*100:.1f}% variance")
        self._plot_pca_variance(pca.explained_variance_ratio_)
        return self

    def _plot_pca_variance(self, ev_ratio):
        cumvar = ev_ratio.cumsum() * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cumvar) + 1), cumvar, color="#4C72B0", linewidth=1.5)
        ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.15, color="#4C72B0")
        for thresh in [80, 90, 95]:
            idx = np.searchsorted(cumvar, thresh)
            ax.axhline(thresh, linestyle="--", linewidth=0.9, color="gray")
            ax.text(len(cumvar) * 0.6, thresh + 0.5, f"{thresh}% @ PC{idx+1}",
                    fontsize=8, color="gray")
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative Explained Variance (%)")
        ax.set_title("PCA — Cumulative Explained Variance")
        fig.tight_layout()
        fig.savefig(FIGURES / "pca_explained_variance.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def find_optimal_k(self, k_range=range(2, 11)):
        print("[Clustering] Finding optimal K (elbow + silhouette)...")
        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(self.pca_coords)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(self.pca_coords, lbl,
                                                sample_size=min(1000, len(lbl))))
        self._plot_elbow(k_range, inertias, silhouettes)
        best_k = list(k_range)[int(np.argmax(silhouettes))]
        print(f"  Best K by silhouette: {best_k}  "
              f"(score = {max(silhouettes):.4f})")
        return best_k

    def _plot_elbow(self, k_range, inertias, silhouettes):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ks = list(k_range)

        ax1.plot(ks, inertias, marker="o", color="#4C72B0", linewidth=1.5)
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Curve")

        ax2.plot(ks, silhouettes, marker="o", color="#DD5500", linewidth=1.5)
        ax2.set_xlabel("Number of Clusters (K)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score by K")
        best_idx = int(np.argmax(silhouettes))
        ax2.scatter([ks[best_idx]], [silhouettes[best_idx]],
                    color="red", zorder=5, s=80, label=f"Best K={ks[best_idx]}")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(FIGURES / "clustering_elbow_silhouette.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def fit(self, k=None):
        if k is None:
            k = self.find_optimal_k()
        self.k = k
        print(f"[Clustering] Fitting KMeans with K={k}...")
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        self.labels = self.kmeans.fit_predict(self.pca_coords)

        sil = silhouette_score(self.pca_coords, self.labels,
                               sample_size=min(1000, len(self.labels)))
        db  = davies_bouldin_score(self.pca_coords, self.labels)
        print(f"  Silhouette score   : {sil:.4f}")
        print(f"  Davies-Bouldin     : {db:.4f}")

        sizes = np.bincount(self.labels)
        for i, s in enumerate(sizes):
            print(f"  Cluster {i}: {s} vehicles ({s/len(self.labels)*100:.1f}%)")

        joblib.dump(self.kmeans, MODELS / "vehicle_kmeans.pkl")
        return self

    # ------------------------------------------------------------------
    def run_umap(self):
        print("[Clustering] Running UMAP for visualization...")
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=20, min_dist=0.1, metric="jaccard")
        self.umap_coords = reducer.fit_transform(self.X)
        self._plot_umap()
        return self

    def _plot_umap(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        coords = self.umap_coords

        # — coloured by cluster
        ax = axes[0]
        for cid in range(self.k):
            mask = self.labels == cid
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=12, alpha=0.65, label=f"Cluster {cid}",
                       color=PALETTE[cid % len(PALETTE)])
        ax.set_title("UMAP — Vehicle Clusters")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=2, fontsize=8)

        # — coloured by task count
        ax2 = axes[1]
        task_counts = self.dl.raw_df.groupby("SPNR8")["CHARACTERISTIC"].nunique()
        tc = task_counts.reindex(self.dl.vehicle_ids).values.astype(float)
        sc = ax2.scatter(coords[:, 0], coords[:, 1],
                         c=tc, cmap="viridis", s=12, alpha=0.7)
        plt.colorbar(sc, ax=ax2, label="Tasks per Vehicle")
        ax2.set_title("UMAP — Coloured by Task Count")
        ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")

        fig.tight_layout()
        fig.savefig(FIGURES / "umap_vehicle_clusters.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def plot_cluster_task_profiles(self):
        """For each cluster, show the most characteristic tasks (high avg frequency)."""
        task_matrix = self.dl.vehicle_tasks
        task_cols   = self.dl.task_cols
        k           = self.k

        fig, axes = plt.subplots(1, k, figsize=(5 * k, 5), sharey=False)
        if k == 1:
            axes = [axes]

        for cid in range(k):
            mask = self.labels == cid
            cluster_freq = task_matrix[mask].mean(axis=0) * 100
            global_freq  = task_matrix.mean(axis=0) * 100
            lift = cluster_freq - global_freq
            top10_idx = np.argsort(lift)[::-1][:10]

            ax = axes[cid]
            bars = ax.barh(
                [task_cols[i][:18] for i in top10_idx[::-1]],
                lift[top10_idx[::-1]],
                color=PALETTE[cid % len(PALETTE)], alpha=0.8
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"Cluster {cid} (n={mask.sum()})\nTop Tasks vs Global", fontsize=9)
            ax.set_xlabel("Frequency Lift (pp)")
            ax.tick_params(axis="y", labelsize=7)

        fig.tight_layout()
        fig.savefig(FIGURES / "cluster_task_profiles.png", dpi=150)
        plt.close(fig)
        print("[Clustering] Cluster task profile plot saved.")

    def get_cluster_labels(self):
        return self.labels
