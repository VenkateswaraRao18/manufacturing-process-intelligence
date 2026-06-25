"""
Streamlit Dashboard — Manufacturing Process Intelligence
=========================================================
Runs locally:  streamlit run app/dashboard.py
Runs on cloud: Deploy from GitHub — no CSV or pkl files needed.
               All data pre-computed in outputs/data/ (~6 MB).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
DATA = ROOT / "outputs" / "data"
FIGS = ROOT / "outputs" / "figures"
REP  = ROOT / "outputs" / "reports"

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Process Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load pre-computed data (cached) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading pre-computed data...")
def load_data():
    vehicle_ids    = np.load(DATA / "vehicle_ids.npy")
    vehicle_config = np.load(DATA / "vehicle_config.npy")
    vehicle_tasks  = np.load(DATA / "vehicle_tasks.npy")
    cluster_labels = np.load(DATA / "cluster_labels.npy")
    pred_probs     = np.load(DATA / "pred_probs.npy")
    tasks_per_v    = np.load(DATA / "tasks_per_vehicle.npy")

    with open(DATA / "task_cols.json")   as f: task_cols   = json.load(f)
    with open(DATA / "config_cols.json") as f: config_cols = json.load(f)
    with open(REP  / "anomaly_report.json")       as f: anomaly  = json.load(f)
    with open(REP  / "multilabel_metrics.json")   as f: metrics  = json.load(f)

    return {
        "vehicle_ids":    vehicle_ids,
        "vehicle_config": vehicle_config,
        "vehicle_tasks":  vehicle_tasks,
        "cluster_labels": cluster_labels,
        "pred_probs":     pred_probs,
        "tasks_per_v":    tasks_per_v,
        "task_cols":      task_cols,
        "config_cols":    config_cols,
        "anomaly":        anomaly,
        "metrics":        metrics,
        "n_clusters":     int(cluster_labels.max()) + 1,
    }


def check_data_ready():
    required = [
        DATA / "vehicle_ids.npy", DATA / "pred_probs.npy",
        DATA / "cluster_labels.npy", REP / "anomaly_report.json",
    ]
    return all(p.exists() for p in required)


# ── Guard ──────────────────────────────────────────────────────────────
if not check_data_ready():
    st.error("Pre-computed data not found.")
    st.markdown("Run the precompute script first:")
    st.code("python scripts/precompute.py", language="bash")
    st.stop()

d = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("Manufacturing Process Intelligence")
st.sidebar.markdown("*Predictive Task Assignment & Anomaly Detection*")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["KPI Overview", "Vehicle Explorer", "Cluster Explorer", "Anomaly Leaderboard"],
)
st.sidebar.divider()
st.sidebar.caption(
    "Pre-computed from 143K-row assembly line dataset. "
    "Run `python main.py && python scripts/precompute.py` to refresh."
)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — KPI Overview
# ══════════════════════════════════════════════════════════════════════
if page == "KPI Overview":
    st.title("KPI Overview")
    st.markdown("End-to-end metrics from the Manufacturing Process Intelligence pipeline.")

    # Dataset metrics
    st.subheader("Dataset")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Vehicles",      f"{len(d['vehicle_ids']):,}")
    c2.metric("Config Features",     "512")
    c3.metric("Tasks in Model",      f"{len(d['task_cols'])}")
    c4.metric("Avg Tasks / Vehicle", f"{d['tasks_per_v'].mean():.1f}")
    c5.metric("Label Density",       "9.84%")

    st.divider()

    # Model metrics
    st.subheader("Model Performance — XGBoost Binary Relevance (325 labels)")
    m = d["metrics"].get("XGBoost", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Micro Precision",       f"{m.get('precision_micro',0)*100:.1f}%",
              help="When model predicts a task, how often it is correct")
    c2.metric("Micro F1 (threshold=0.20)", "28.4%",
              help="+40% vs default 0.5 threshold")
    c3.metric("Hamming Loss",          f"{m.get('hamming_loss',0):.4f}",
              help="Lower is better — fraction of labels wrong")
    c4.metric("Label Ranking Avg Precision", f"{m.get('lrap',0):.4f}",
              help="Each true task ranked in top 32% of 325 labels")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Clustering — 10 Variant Families")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters (K)",         str(d["n_clusters"]))
        c2.metric("Silhouette Score",      "0.38")
        c3.metric("PCA Variance Retained", "94.2%")

        sizes = pd.Series(d["cluster_labels"]).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        palette = sns.color_palette("tab10", d["n_clusters"])
        ax.bar(sizes.index.astype(str), sizes.values, color=palette)
        ax.set_xlabel("Cluster ID"); ax.set_ylabel("Vehicles")
        ax.set_title("Vehicles per Cluster")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col_b:
        st.subheader("Anomaly Detection")
        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged Vehicles (P95)", d["anomaly"]["flagged_p95"])
        c2.metric("Flag Rate",              f"{d['anomaly']['flagged_pct']}%")
        c3.metric("Mean Deviation",         f"{d['anomaly']['mean_deviation_tasks']} tasks")

        top5 = d["anomaly"]["top_20_anomalous_vehicles"][:5]
        df5  = pd.DataFrame(top5)[["vehicle_id","combined_score","deviation_tasks",
                                    "unexpected_tasks","missed_tasks"]]
        df5.columns = ["Vehicle ID","Score","Deviation","Unexpected","Missed"]
        st.markdown("**Top 5 Most Anomalous Vehicles**")
        st.dataframe(df5, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Analysis Plots")
    tabs = st.tabs(["EDA", "Clustering", "Model", "Anomaly"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.image(str(FIGS/"tasks_per_vehicle_dist.png"),    caption="Tasks per Vehicle")
        c2.image(str(FIGS/"task_frequency_distribution.png"),caption="Task Frequency Distribution")
        c1.image(str(FIGS/"config_feature_frequency.png"),  caption="Config Feature Prevalence")
        c2.image(str(FIGS/"task_cooccurrence_heatmap.png"), caption="Task Co-occurrence (Jaccard)")

    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.image(str(FIGS/"pca_explained_variance.png"),       caption="PCA Explained Variance")
        c2.image(str(FIGS/"clustering_elbow_silhouette.png"),  caption="Elbow + Silhouette")
        c1.image(str(FIGS/"umap_vehicle_clusters.png"),        caption="UMAP Embedding")
        c2.image(str(FIGS/"cluster_task_profiles.png"),        caption="Cluster Task Profiles")

    with tabs[2]:
        c1, c2 = st.columns(2)
        c1.image(str(FIGS/"multilabel_model_comparison.png"), caption="Model Comparison")
        c2.image(str(FIGS/"per_label_f1_distribution.png"),   caption="Per-label F1")
        c1.image(str(FIGS/"threshold_tuning.png"),            caption="Threshold Tuning")

    with tabs[3]:
        c1, c2 = st.columns(2)
        c1.image(str(FIGS/"anomaly_score_distributions.png"),    caption="Score Distributions")
        c2.image(str(FIGS/"anomaly_deviation_vs_combined.png"),  caption="Deviation vs Combined")
        c1.image(str(FIGS/"anomaly_task_signatures.png"),        caption="Anomaly Task Signatures")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Vehicle Explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "Vehicle Explorer":
    st.title("Vehicle Explorer")
    st.markdown("Select a vehicle to see its config profile, predicted tasks, and anomaly status.")

    selected_vid = st.selectbox(
        "Select Vehicle Serial Number (SPNR8)",
        sorted(d["vehicle_ids"].tolist())
    )
    threshold = st.slider("Decision Threshold", 0.10, 0.80, 0.20, 0.05,
                          help="Lower = more tasks predicted")

    idx = int(np.where(d["vehicle_ids"] == selected_vid)[0][0])
    probs       = d["pred_probs"][idx]
    config_vec  = d["vehicle_config"][idx]
    actual_mask = d["vehicle_tasks"][idx].astype(bool)
    actual_set  = set(np.array(d["task_cols"])[actual_mask])

    pred_mask  = probs >= threshold
    pred_set   = set(np.array(d["task_cols"])[pred_mask])
    true_pos   = pred_set & actual_set
    false_pos  = pred_set - actual_set
    false_neg  = actual_set - pred_set

    prec = len(true_pos) / len(pred_set)  if pred_set  else 0.0
    rec  = len(true_pos) / len(actual_set) if actual_set else 0.0
    f1   = 2 * prec * rec / (prec + rec)  if (prec + rec) > 0 else 0.0

    # Anomaly lookup
    anomaly_entry = next(
        (e for e in d["anomaly"]["top_20_anomalous_vehicles"] if e["vehicle_id"] == selected_vid),
        None
    )

    st.divider()
    col_info, col_pred = st.columns([1, 2])

    with col_info:
        st.subheader("Vehicle Profile")
        st.metric("Config Features ON",  f"{int(config_vec.sum())} / 512  ({config_vec.mean()*100:.1f}%)")
        st.metric("Actual Tasks (model labels)", int(actual_mask.sum()))
        st.metric("Cluster Assignment",  f"Cluster {d['cluster_labels'][idx]}")
        if anomaly_entry:
            st.metric("Anomaly Score", f"{anomaly_entry['combined_score']:.3f}",
                      delta=f"{anomaly_entry['deviation_tasks']} task deviation",
                      delta_color="inverse")
        else:
            st.metric("Anomaly Score", "Normal")

    with col_pred:
        st.subheader("Predicted vs Actual Tasks")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted",    len(pred_set))
        c2.metric("Correct (TP)", len(true_pos))
        c3.metric("Over-predicted (FP)", len(false_pos))
        c4.metric("Missed (FN)",  len(false_neg))

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{prec*100:.1f}%")
        c2.metric("Recall",    f"{rec*100:.1f}%")
        c3.metric("F1",        f"{f1*100:.1f}%")

    st.divider()
    st.subheader("Top 25 Task Probabilities")
    top25_idx = np.argsort(probs)[::-1][:25]
    task_arr  = np.array(d["task_cols"])
    prob_df   = pd.DataFrame({
        "Task Code":   task_arr[top25_idx],
        "Probability": probs[top25_idx].round(4),
        "Predicted":   ["Yes" if probs[i] >= threshold else "No" for i in top25_idx],
        "Actual":      ["Yes" if task_arr[i] in actual_set  else "No" for i in top25_idx],
    })

    def row_color(row):
        color = "#d4edda" if row["Predicted"] == row["Actual"] else "#f8d7da"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        prob_df.style.apply(row_color, axis=1),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Cluster Explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "Cluster Explorer":
    st.title("Cluster Explorer")
    st.markdown("Explore the 10 vehicle variant families discovered by PCA + KMeans.")

    selected_cluster = st.selectbox(
        "Select Cluster", list(range(d["n_clusters"])),
        format_func=lambda x: f"Cluster {x}"
    )

    mask       = d["cluster_labels"] == selected_cluster
    n_in       = int(mask.sum())
    task_arr   = np.array(d["task_cols"])
    cfg_arr    = np.array(d["config_cols"])

    st.markdown(f"**{n_in} vehicles** ({n_in/len(d['cluster_labels'])*100:.1f}% of fleet)")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Task Signature (Lift over Global Avg)")
        cluster_freq = d["vehicle_tasks"][mask].mean(axis=0) * 100
        global_freq  = d["vehicle_tasks"].mean(axis=0) * 100
        lift = cluster_freq - global_freq
        top10 = np.argsort(lift)[::-1][:10]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([task_arr[i][:20] for i in top10[::-1]], lift[top10[::-1]],
                color="#4C72B0", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Frequency Lift (pp vs global)")
        ax.set_title(f"Cluster {selected_cluster} — Signature Tasks")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.subheader("Distinguishing Config Features")
        cluster_cfg  = d["vehicle_config"][mask].mean(axis=0) * 100
        global_cfg   = d["vehicle_config"].mean(axis=0) * 100
        cfg_lift     = cluster_cfg - global_cfg
        top10_cfg    = np.argsort(np.abs(cfg_lift))[::-1][:10]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ["#DD4444" if cfg_lift[i] > 0 else "#4C72B0" for i in top10_cfg[::-1]]
        ax2.barh([cfg_arr[i] for i in top10_cfg[::-1]], cfg_lift[top10_cfg[::-1]],
                 color=colors, alpha=0.85)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Feature Frequency Lift (pp)")
        ax2.set_title(f"Cluster {selected_cluster} — Config Features")
        ax2.tick_params(axis="y", labelsize=8)
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.divider()
    st.subheader("Vehicles in this Cluster")
    cluster_vids = d["vehicle_ids"][mask]
    cluster_tasks = d["tasks_per_v"][mask]
    cluster_df = pd.DataFrame({
        "Vehicle ID":     cluster_vids,
        "Tasks Recorded": cluster_tasks,
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Vehicles Flagged (P95)", d["anomaly"]["flagged_p95"])
    c2.metric("Flag Rate",             f"{d['anomaly']['flagged_pct']}%")
    c3.metric("Avg Deviation",         f"{d['anomaly']['mean_deviation_tasks']} tasks")

    top20 = d["anomaly"]["top_20_anomalous_vehicles"]
    df    = pd.DataFrame(top20)
    df.columns = ["Vehicle ID","Combined Score","Deviation","Unexpected Tasks","Missed Tasks"]

    st.divider()
    st.subheader("Top 20 Most Anomalous Vehicles")
    st.dataframe(
        df.style
          .background_gradient(subset=["Combined Score"], cmap="Reds")
          .background_gradient(subset=["Deviation"],      cmap="Oranges")
          .format({"Combined Score": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Unexpected vs Missed Tasks")
        fig, ax = plt.subplots(figsize=(5, 4))
        vids = df["Vehicle ID"].astype(str)
        ax.barh(vids[::-1], df["Unexpected Tasks"][::-1],
                color="#DD8800", alpha=0.8, label="Unexpected")
        ax.barh(vids[::-1], df["Missed Tasks"][::-1],
                left=df["Unexpected Tasks"][::-1],
                color="#4C72B0", alpha=0.8, label="Missed")
        ax.set_xlabel("Task Count")
        ax.legend(fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown("**Interpreting anomaly types:**")
        st.markdown("""
| Pattern | Likely Cause |
|---|---|
| Many **missed** tasks | Exited line early or incomplete process recording |
| Many **unexpected** tasks | Undocumented variant or reconfiguration mid-build |
| Both high | Rework event or significant process deviation |
        """)
        st.image(str(FIGS / "anomaly_deviation_vs_combined.png"),
                 caption="Deviation vs Combined Score — all 2,869 vehicles")
