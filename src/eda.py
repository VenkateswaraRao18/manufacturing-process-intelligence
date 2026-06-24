import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

FIGURES = Path(__file__).parent.parent / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


class EDAAnalyzer:
    def __init__(self, data_loader):
        self.dl = data_loader

    def run_all(self):
        print("\n=== EDA ===")
        self.plot_tasks_per_vehicle()
        self.plot_task_frequency_distribution()
        self.plot_config_feature_density()
        self.plot_config_feature_frequency()
        self.plot_label_cooccurrence_heatmap()
        self.plot_tasks_per_vehicle_violin()
        self.print_key_findings()
        print(f"\nAll EDA plots saved to {FIGURES}")

    # ------------------------------------------------------------------
    def plot_tasks_per_vehicle(self):
        raw = self.dl.raw_df
        tasks_per_v = raw.groupby("SPNR8")["CHARACTERISTIC"].nunique()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(tasks_per_v, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.axvline(tasks_per_v.median(), color="#DD4444", linestyle="--",
                   linewidth=1.5, label=f"Median = {tasks_per_v.median():.0f}")
        ax.axvline(tasks_per_v.mean(), color="#22AA44", linestyle="--",
                   linewidth=1.5, label=f"Mean = {tasks_per_v.mean():.1f}")
        ax.set_xlabel("Number of Tasks per Vehicle")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Distribution of Tasks per Vehicle")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "tasks_per_vehicle_dist.png", dpi=150)
        plt.close(fig)

    def plot_tasks_per_vehicle_violin(self):
        raw = self.dl.raw_df
        tasks_per_v = raw.groupby("SPNR8")["CHARACTERISTIC"].nunique()

        fig, ax = plt.subplots(figsize=(5, 6))
        ax.violinplot(tasks_per_v, showmedians=True)
        ax.set_ylabel("Tasks per Vehicle")
        ax.set_title("Task Count Spread across Vehicles")
        ax.set_xticks([])
        fig.tight_layout()
        fig.savefig(FIGURES / "tasks_per_vehicle_violin.png", dpi=150)
        plt.close(fig)

    def plot_task_frequency_distribution(self):
        raw = self.dl.raw_df
        vehicles_per_task = raw.groupby("CHARACTERISTIC")["SPNR8"].nunique()
        n_vehicles = raw["SPNR8"].nunique()
        pct = (vehicles_per_task / n_vehicles) * 100

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].hist(pct, bins=50, color="#DD8800", edgecolor="white", linewidth=0.4)
        axes[0].set_xlabel("% of Vehicles that Perform the Task")
        axes[0].set_ylabel("Number of Tasks")
        axes[0].set_title("Task Frequency Distribution")
        for thresh, color in [(5, "#CC3333"), (10, "#3333CC")]:
            cnt = (pct >= thresh).sum()
            axes[0].axvline(thresh, color=color, linestyle="--", linewidth=1.2,
                            label=f">={thresh}%: {cnt} tasks")
        axes[0].legend(fontsize=9)

        # Cumulative
        sorted_pct = np.sort(pct)[::-1]
        axes[1].plot(np.arange(1, len(sorted_pct) + 1), sorted_pct,
                     color="#4C72B0", linewidth=1.5)
        axes[1].axhline(5, color="#CC3333", linestyle="--", linewidth=1, label="5% threshold")
        axes[1].set_xlabel("Task Rank (by frequency)")
        axes[1].set_ylabel("% Vehicles")
        axes[1].set_title("Task Frequency (Sorted)")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(FIGURES / "task_frequency_distribution.png", dpi=150)
        plt.close(fig)

    def plot_config_feature_density(self):
        config = self.dl.vehicle_config
        density = config.mean(axis=1) * 100  # % features = 1 per vehicle

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(density, bins=35, color="#55AA88", edgecolor="white", linewidth=0.5)
        ax.axvline(density.mean(), color="#CC4444", linestyle="--",
                   linewidth=1.5, label=f"Mean = {density.mean():.1f}%")
        ax.set_xlabel("% Config Features Set to 1 per Vehicle")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Vehicle Config Density Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "config_density_distribution.png", dpi=150)
        plt.close(fig)

    def plot_config_feature_frequency(self):
        config = self.dl.vehicle_config
        feature_freq = config.mean(axis=0) * 100  # % vehicles that have each feature ON
        sorted_ff = np.sort(feature_freq)[::-1]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sorted_ff, color="#7744BB", linewidth=1.5)
        ax.fill_between(np.arange(len(sorted_ff)), sorted_ff, alpha=0.2, color="#7744BB")
        ax.set_xlabel("Config Feature Rank (by frequency)")
        ax.set_ylabel("% Vehicles with Feature = 1")
        ax.set_title("Config Feature Prevalence across Vehicles")
        ax.axhline(50, color="#CC4444", linestyle="--", linewidth=1, label="50%")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "config_feature_frequency.png", dpi=150)
        plt.close(fig)

    def plot_label_cooccurrence_heatmap(self):
        # Top 30 most frequent tasks
        task_matrix = self.dl.vehicle_tasks
        task_freq = task_matrix.mean(axis=0)
        top30_idx = np.argsort(task_freq)[::-1][:30]
        top30_matrix = task_matrix[:, top30_idx].astype(float)
        top30_labels = [self.dl.task_cols[i][:12] for i in top30_idx]

        # Jaccard co-occurrence
        n = top30_matrix.shape[1]
        cooc = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                inter = (top30_matrix[:, i] * top30_matrix[:, j]).sum()
                union = ((top30_matrix[:, i] + top30_matrix[:, j]) >= 1).sum()
                cooc[i, j] = inter / union if union > 0 else 0

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cooc, xticklabels=top30_labels, yticklabels=top30_labels,
                    cmap="YlOrRd", ax=ax, linewidths=0.3, square=True,
                    cbar_kws={"label": "Jaccard Similarity"})
        ax.set_title("Task Co-occurrence (Jaccard) — Top 30 Tasks")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / "task_cooccurrence_heatmap.png", dpi=150)
        plt.close(fig)

    def print_key_findings(self):
        raw = self.dl.raw_df
        n_v = raw["SPNR8"].nunique()
        n_t = raw["CHARACTERISTIC"].nunique()
        tasks_per_v = raw.groupby("SPNR8")["CHARACTERISTIC"].nunique()
        vehicles_per_task = raw.groupby("CHARACTERISTIC")["SPNR8"].nunique()
        config_density = self.dl.vehicle_config.mean(axis=1) * 100

        print("\n--- Key EDA Findings ---")
        print(f"Vehicles: {n_v:,}  |  Unique tasks: {n_t:,}")
        print(f"Tasks/vehicle: {tasks_per_v.mean():.1f} mean, "
              f"{tasks_per_v.median():.0f} median, "
              f"{tasks_per_v.std():.1f} std")
        print(f"Vehicles/task: {vehicles_per_task.mean():.1f} mean, "
              f"{vehicles_per_task.median():.0f} median")
        print(f"Tasks in >50% vehicles: {(vehicles_per_task / n_v > 0.5).sum()}")
        print(f"Tasks in <5% vehicles : {(vehicles_per_task / n_v < 0.05).sum()}")
        print(f"Config density: {config_density.mean():.1f}% avg features ON per vehicle")
        print(f"Config density range  : [{config_density.min():.1f}%, {config_density.max():.1f}%]")
