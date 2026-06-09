import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"

LOG_DIR = os.path.join(BASE_DIR, "bibm2026_project/logs")
FIG_DIR = os.path.join(BASE_DIR, "bibm2026_project/figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.10)

def read_if_exists(filename):
    path = os.path.join(LOG_DIR, filename)
    if os.path.exists(path):
        print("Reading:", path)
        return pd.read_csv(path)
    print("Missing:", path)
    return pd.DataFrame()

frames = []

baseline = read_if_exists("baseline_model_metrics.csv")
if not baseline.empty:
    baseline["task"] = "LUAD driver prediction"
    frames.append(baseline)

mlp = read_if_exists("MLP_LUAD_metrics.csv")
if not mlp.empty:
    mlp["task"] = "LUAD driver prediction"
    frames.append(mlp)

resmlp = read_if_exists("ResMLP_LUAD_metrics.csv")
if not resmlp.empty:
    resmlp["task"] = "LUAD driver prediction"
    frames.append(resmlp)

qc = read_if_exists("QC_model_metrics.csv")
if not qc.empty:
    qc["task"] = "SeqQC confidence prediction"
    qc["feature_set"] = "QC"
    frames.append(qc)

mvf = read_if_exists("MVF_ablation_metrics_rounded.csv")
if not mvf.empty:
    if "task" not in mvf.columns:
        mvf["task"] = "LUAD driver prediction"
    frames.append(mvf)

cross = read_if_exists("coad_gbm_cross_cancer_metrics_rounded.csv")
if not cross.empty:
    frames.append(cross)

if not frames:
    raise RuntimeError("No metric files found. Please run previous scripts first.")

all_metrics = pd.concat(frames, ignore_index=True)

# Standardize columns
for col in ["task", "model", "feature_set", "AUROC", "AUPRC", "F1"]:
    if col not in all_metrics.columns:
        all_metrics[col] = None

all_metrics = all_metrics[["task", "model", "feature_set", "AUROC", "AUPRC", "F1"]]

# Remove misleading rows: no LUAD + QC fusion claim
bad_mask = (
    (all_metrics["task"].astype(str) == "LUAD driver prediction") &
    (all_metrics["model"].astype(str) == "MVF") &
    (all_metrics["feature_set"].astype(str).str.contains("QC", na=False))
)
all_metrics = all_metrics[~bad_mask].copy()

# Save combined final tables
all_metrics.to_csv(os.path.join(LOG_DIR, "final_combined_metrics.csv"), index=False)
all_metrics.round(4).to_csv(os.path.join(LOG_DIR, "final_combined_metrics_rounded.csv"), index=False)

all_metrics.to_csv(os.path.join(LOG_DIR, "BIBM_final_metrics_clean.csv"), index=False)
all_metrics.round(4).to_csv(os.path.join(LOG_DIR, "BIBM_final_metrics_clean_rounded.csv"), index=False)

print("\nFINAL COMBINED METRICS")
print(all_metrics.round(4))

base_colors = {
    "RandomForest": "#1f77b4",
    "LightGBM": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "MLP": "#d62728",
    "ResMLP": "#9467bd",
    "MVF": "#8c564b",
    "QC-MLP": "#e377c2"
}

def make_palette(models):
    fallback = sns.color_palette("tab10", n_colors=max(len(models), 3)).as_hex()
    palette = {}
    for i, m in enumerate(models):
        palette[m] = base_colors.get(m, fallback[i % len(fallback)])
    return palette

# Paper-ready bar plots by task
for task in all_metrics["task"].dropna().unique():
    df_task = all_metrics[all_metrics["task"] == task].copy()
    df_task["plot_label"] = (
        df_task["model"].astype(str) + "\n" + df_task["feature_set"].astype(str)
    )

    palette = make_palette(df_task["model"].astype(str).unique())

    for metric in ["AUROC", "AUPRC", "F1"]:
        df_plot = df_task.dropna(subset=[metric]).sort_values(metric, ascending=False)
        if df_plot.empty:
            continue

        plt.figure(figsize=(13, 6))
        ax = sns.barplot(
            data=df_plot,
            x="plot_label",
            y=metric,
            hue="model",
            dodge=False,
            palette=palette
        )

        ax.set_title(f"{task}: {metric}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Model", frameon=True, fontsize=8)

        plt.tight_layout()

        safe_task = task.replace(" ", "_").replace("/", "_")
        out1 = os.path.join(FIG_DIR, f"paper_{safe_task}_{metric}.png")
        out2 = os.path.join(FIG_DIR, f"BIBM_{safe_task}_{metric}.png")

        plt.savefig(out1, dpi=300, bbox_inches="tight")
        plt.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close()

print("\nSaved:")
print(os.path.join(LOG_DIR, "final_combined_metrics_rounded.csv"))
print(os.path.join(LOG_DIR, "BIBM_final_metrics_clean_rounded.csv"))
print("Figures saved in:", FIG_DIR)