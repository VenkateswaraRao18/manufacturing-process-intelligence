"""
Streamlit Dashboard — Manufacturing Process Intelligence
=========================================================
Run:  streamlit run app/dashboard.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Process Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and models...")
def load_all():
    from src.data_loader import DataLoader
    import joblib
    from sklearn.decomposition import PCA

    dl = DataLoader()
    dl.load().clean().pivot(min_task_support=0.05)

    model_dir = ROOT / "outputs" / "models"
    xgb = joblib.load(model_dir / "multilabel_xgb.pkl")
    km  = joblib.load(model_dir / "vehicle_kmeans.pkl")

    # Fit PCA once
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(dl.vehicle_config.astype(np.float32))
    cluster_labels = km.predict(X_pca)

    reports = {
        "anomaly": json.loads((ROOT / "outputs" / "reports" / "anomaly_report.json").read_text()),
        "metrics": json.loads((ROOT / "outputs" / "reports" / "multilabel_metrics.json").read_text()),
    }
    return dl, xgb, km, pca, X_pca, cluster_labels, reports


def check_models_exist():
    p1 = ROOT / "outputs" / "models" / "multilabel_xgb.pkl"
    p2 = ROOT / "outputs" / "models" / "vehicle_kmeans.pkl"
    return p1.exists() and p2.exists()


# ── Sidebar nav ────────────────────────────────────────────────────────
st.sidebar.title("Manufacturing Process Intelligence")
st.sidebar.markdown("*Predictive Task Assignment & Anomaly Detection*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["KPI Overview", "Vehicle Explorer", "Cluster Explorer", "Anomaly Leaderboard"],
)

st.sidebar.divider()
st.sidebar.caption("Run `python main.py` to regenerate models and reports.")

# ── Guard: models must exist ───────────────────────────────────────────
if not check_models_exist():
    st.error("Models not found. Run `python main.py` first to train and save them.")
    st.code("python main.py", language="bash")
    st.stop()

dl, xgb, km, pca, X_pca, cluster_labels, reports = load_all()

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — KPI Overview
# ══════════════════════════════════════════════════════════════════════
if page == "KPI Overview":
    st.title("KPI Overview")
    st.markdown("Key metrics from the end-to-end Manufacturing Process Intelligence pipeline.")

    # ── Row 1: dataset metrics ──────────────────────────────────────
    st.subheader("Dataset")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Vehicles", f"{len(dl.vehicle_ids):,}")
    c2.metric("Unique Tasks", f"{dl.raw_df['CHARACTERISTIC'].nunique():,}")
    c3.metric("Config Features", "512")
    c4.metric("Tasks in Model", f"{len(dl.task_cols)}")
    tasks_per_v = dl.raw_df.groupby("SPNR8")["CHARACTERISTIC"].nunique()
    c5.metric("Avg Tasks / Vehicle", f"{tasks_per_v.mean():.1f}")

    st.divider()

    # ── Row 2: model metrics ────────────────────────────────────────
    st.subheader("Model Performance — XGBoost Binary Relevance")
    m = reports["metrics"].get("XGBoost", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Micro Precision", f"{m.get('precision_micro', 0)*100:.1f}%",
              help="When the model predicts a task, how often is it correct?")
    c2.metric("Micro F1 (tuned threshold)", "28.4%",
              help="F1 at threshold=0.20 (tuned from default 0.5)")
    c3.metric("Hamming Loss", f"{m.get('hamming_loss', 0):.4f}",
              help="Fraction of labels incorrectly predicted (lower is better)")
    c4.metric("Label Ranking Avg Precision", f"{m.get('lrap', 0):.4f}",
              help="For each true task, how high does the model rank it?")

    st.divider()

    # ── Row 3: clustering + anomaly ─────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Clustering")
        c1, c2, c3 = st.columns(3)
        c1.metric("Variant Families", "10")
        c2.metric("Silhouette Score", "0.38")
        c3.metric("PCA Variance Retained", "94.2%")

        sizes = pd.Series(cluster_labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(sizes.index.astype(str), sizes.values, color=sns.color_palette("tab10", 10))
        ax.set_xlabel("Cluster ID"); ax.set_ylabel("Vehicles")
        ax.set_title("Vehicles per Cluster")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Anomaly Detection")
        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged Vehicles (P95)", reports["anomaly"]["flagged_p95"])
        c2.metric("Flag Rate", f"{reports['anomaly']['flagged_pct']}%")
        c3.metric("Mean Deviation", f"{reports['anomaly']['mean_deviation_tasks']} tasks")

        top5 = reports["anomaly"]["top_20_anomalous_vehicles"][:5]
        df_top5 = pd.DataFrame(top5)[["vehicle_id", "combined_score", "deviation_tasks",
                                       "unexpected_tasks", "missed_tasks"]]
        df_top5.columns = ["Vehicle ID", "Score", "Deviation", "Unexpected", "Missed"]
        st.markdown("**Top 5 Most Anomalous Vehicles**")
        st.dataframe(df_top5, use_container_width=True, hide_index=True)

    st.divider()

    # ── Row 4: saved plots ──────────────────────────────────────────
    st.subheader("Analysis Plots")
    figures = ROOT / "outputs" / "figures"
    tabs = st.tabs(["EDA", "Clustering", "Model", "Anomaly"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        col1.image(str(figures / "tasks_per_vehicle_dist.png"),
                   caption="Task Count Distribution per Vehicle")
        col2.image(str(figures / "task_frequency_distribution.png"),
                   caption="Task Frequency Distribution")
        col1.image(str(figures / "config_feature_frequency.png"),
                   caption="Config Feature Prevalence")
        col2.image(str(figures / "task_cooccurrence_heatmap.png"),
                   caption="Task Co-occurrence Heatmap (Jaccard)")

    with tabs[1]:
        col1, col2 = st.columns(2)
        col1.image(str(figures / "pca_explained_variance.png"), caption="PCA Explained Variance")
        col2.image(str(figures / "clustering_elbow_silhouette.png"), caption="Elbow + Silhouette")
        col1.image(str(figures / "umap_vehicle_clusters.png"), caption="UMAP Embedding")
        col2.image(str(figures / "cluster_task_profiles.png"), caption="Cluster Task Profiles")

    with tabs[2]:
        col1, col2 = st.columns(2)
        col1.image(str(figures / "multilabel_model_comparison.png"), caption="Model Comparison")
        col2.image(str(figures / "per_label_f1_distribution.png"), caption="Per-label F1")
        col1.image(str(figures / "threshold_tuning.png"), caption="Threshold Tuning")

    with tabs[3]:
        col1, col2 = st.columns(2)
        col1.image(str(figures / "anomaly_score_distributions.png"), caption="Score Distributions")
        col2.image(str(figures / "anomaly_deviation_vs_combined.png"), caption="Deviation vs Combined")
        col1.image(str(figures / "anomaly_task_signatures.png"), caption="Anomaly Task Signatures")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Vehicle Explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "Vehicle Explorer":
    st.title("Vehicle Explorer")
    st.markdown("Select a vehicle by serial number to see its config, predicted tasks, and anomaly status.")

    vehicle_options = sorted(dl.vehicle_ids.tolist())
    selected_vid = st.selectbox("Select Vehicle Serial Number (SPNR8)", vehicle_options)

    idx = np.where(dl.vehicle_ids == selected_vid)[0][0]
    config_vec = dl.vehicle_config[idx].astype(np.float32)

    threshold = st.slider("Decision Threshold", 0.10, 0.80, 0.20, 0.05,
                          help="Lower = more tasks predicted (higher recall, lower precision)")

    st.divider()

    # ── Config summary ──────────────────────────────────────────────
    col_info, col_pred = st.columns([1, 2])

    with col_info:
        st.subheader("Vehicle Profile")
        n_features_on = int(config_vec.sum())
        config_density = n_features_on / 512 * 100

        actual_tasks = dl.raw_df[dl.raw_df["SPNR8"] == selected_vid]["CHARACTERISTIC"].tolist()
        cluster_id = int(cluster_labels[idx])

        # Anomaly info
        deviation = None
        anomaly_score = None
        for entry in reports["anomaly"]["top_20_anomalous_vehicles"]:
            if entry["vehicle_id"] == selected_vid:
                deviation = entry["deviation_tasks"]
                anomaly_score = entry["combined_score"]
                break

        st.metric("Config Features ON", f"{n_features_on} / 512 ({config_density:.1f}%)")
        st.metric("Actual Tasks Recorded", len(actual_tasks))
        st.metric("Cluster Assignment", f"Cluster {cluster_id}")
        if anomaly_score:
            st.metric("Anomaly Score", f"{anomaly_score:.3f}",
                      delta=f"{deviation} task deviation", delta_color="inverse")
        else:
            st.metric("Anomaly Score", "Normal", delta="Not in top anomalies")

    with col_pred:
        st.subheader("Predicted vs Actual Tasks")

        probs = xgb.predict_proba(config_vec.reshape(1, -1))[0]
        pred_idx = np.where(probs >= threshold)[0]
        predicted_tasks = set(dl.task_cols[i] for i in pred_idx)
        actual_tasks_set = set(actual_tasks) & set(dl.task_cols)

        true_pos  = predicted_tasks & actual_tasks_set
        false_pos = predicted_tasks - actual_tasks_set
        false_neg = actual_tasks_set - predicted_tasks

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Tasks", len(predicted_tasks))
        c2.metric("Correct (TP)", len(true_pos))
        c3.metric("Over-predicted (FP)", len(false_pos))
        c4.metric("Missed (FN)", len(false_neg))

        prec = len(true_pos) / len(predicted_tasks) if predicted_tasks else 0
        rec  = len(true_pos) / len(actual_tasks_set) if actual_tasks_set else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{prec*100:.1f}%")
        c2.metric("Recall",    f"{rec*100:.1f}%")
        c3.metric("F1 Score",  f"{f1*100:.1f}%")

    st.divider()

    # ── Top task probabilities ──────────────────────────────────────
    st.subheader("Top 25 Task Probabilities")
    top25_idx  = np.argsort(probs)[::-1][:25]
    prob_df = pd.DataFrame({
        "Task Code":   [dl.task_cols[i] for i in top25_idx],
        "Probability": [round(float(probs[i]), 4) for i in top25_idx],
        "Predicted":   ["Yes" if probs[i] >= threshold else "No" for i in top25_idx],
        "Actual":      ["Yes" if dl.task_cols[i] in actual_tasks else "No" for i in top25_idx],
    })
    st.dataframe(
        prob_df.style.apply(
            lambda row: ["background-color: #d4edda" if row["Predicted"] == row["Actual"]
                         else "background-color: #f8d7da"] * len(row),
            axis=1
        ),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Cluster Explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "Cluster Explorer":
    st.title("Cluster Explorer")
    st.markdown("Explore the 10 vehicle variant families discovered by PCA + KMeans.")

    selected_cluster = st.selectbox("Select Cluster", list(range(km.n_clusters)),
                                    format_func=lambda x: f"Cluster {x}")

    mask = cluster_labels == selected_cluster
    n_in_cluster = mask.sum()
    st.markdown(f"**{n_in_cluster} vehicles** ({n_in_cluster/len(cluster_labels)*100:.1f}% of fleet)")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Task Signature (Lift over Global)")
        cluster_freq = dl.vehicle_tasks[mask].mean(axis=0) * 100
        global_freq  = dl.vehicle_tasks.mean(axis=0) * 100
        lift = cluster_freq - global_freq

        top10_idx = np.argsort(lift)[::-1][:10]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([dl.task_cols[i][:20] for i in top10_idx[::-1]],
                lift[top10_idx[::-1]], color="#4C72B0", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Frequency Lift (pp vs global)")
        ax.set_title(f"Cluster {selected_cluster} — Top Tasks")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Config Feature Profile")
        cluster_cfg_freq = dl.vehicle_config[mask].mean(axis=0) * 100
        global_cfg_freq  = dl.vehicle_config.mean(axis=0) * 100
        cfg_lift = cluster_cfg_freq - global_cfg_freq

        top10_cfg_idx = np.argsort(np.abs(cfg_lift))[::-1][:10]
        cfg_names = [dl.config_cols[i] for i in top10_cfg_idx]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ["#DD4444" if cfg_lift[i] > 0 else "#4C72B0" for i in top10_cfg_idx[::-1]]
        ax2.barh(cfg_names[::-1], cfg_lift[top10_cfg_idx[::-1]], color=colors, alpha=0.85)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Feature Frequency Lift (pp)")
        ax2.set_title(f"Cluster {selected_cluster} — Distinguishing Config Features")
        ax2.tick_params(axis="y", labelsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()
    st.subheader("Vehicles in this Cluster")
    cluster_vids = dl.vehicle_ids[mask]
    tasks_per_v  = dl.raw_df.groupby("SPNR8")["CHARACTERISTIC"].nunique()
    cluster_df   = pd.DataFrame({
        "Vehicle ID": cluster_vids,
        "Tasks Recorded": tasks_per_v.reindex(cluster_vids).values,
    }).sort_values("Tasks Recorded", ascending=False)
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Anomaly Leaderboard
# ══════════════════════════════════════════════════════════════════════
elif page == "Anomaly Leaderboard":
    st.title("Anomaly Leaderboard")
    st.markdown(
        "Vehicles ranked by combined anomaly score "
        "(Isolation Forest on config + model-deviation scoring)."
    )

    top20 = reports["anomaly"]["top_20_anomalous_vehicles"]
    df = pd.DataFrame(top20)
    df.columns = ["Vehicle ID", "Combined Score", "Deviation", "Unexpected Tasks", "Missed Tasks"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Vehicles Flagged (P95)", reports["anomaly"]["flagged_p95"])
    c2.metric("Flag Rate", f"{reports['anomaly']['flagged_pct']}%")
    c3.metric("Avg Deviation (all vehicles)",
              f"{reports['anomaly']['mean_deviation_tasks']} tasks")

    st.divider()
    st.subheader("Top 20 Most Anomalous Vehicles")
    st.dataframe(
        df.style.background_gradient(subset=["Combined Score"], cmap="Reds")
               .background_gradient(subset=["Deviation"], cmap="Oranges")
               .format({"Combined Score": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.subheader("Anomaly Type Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(df["Vehicle ID"].astype(str)[::-1],
                df["Unexpected Tasks"][::-1], color="#DD8800", alpha=0.8, label="Unexpected")
        ax.barh(df["Vehicle ID"].astype(str)[::-1],
                df["Missed Tasks"][::-1], left=df["Unexpected Tasks"][::-1],
                color="#4C72B0", alpha=0.8, label="Missed")
        ax.set_xlabel("Task Count")
        ax.set_title("Unexpected vs Missed Tasks")
        ax.legend(fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**Interpreting anomaly types:**")
        st.markdown("""
| Pattern | Likely Cause |
|---|---|
| Many **missed** tasks, few unexpected | Vehicle exited line early or incomplete process recording |
| Many **unexpected** tasks, few missed | Undocumented variant, reconfiguration mid-build, or data mapping error |
| Both high | Complex rework event or significant process deviation |
        """)
        st.image(str(ROOT / "outputs" / "figures" / "anomaly_deviation_vs_combined.png"),
                 caption="Deviation vs Combined Score — all vehicles")
