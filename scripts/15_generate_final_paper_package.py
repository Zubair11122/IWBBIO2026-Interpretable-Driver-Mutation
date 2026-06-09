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

sns.set_theme(style="whitegrid", font_scale=1.1)

def read_if(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print("Missing:", path)
    return pd.DataFrame()

frames = []

baseline = read_if(os.path.join(LOG_DIR, "baseline_model_metrics.csv"))
if not baseline.empty:
    baseline["task"] = "LUAD driver prediction"
    frames.append(baseline)

mlp = read_if(os.path.join(LOG_DIR, "MLP_LUAD_metrics.csv"))
if not mlp.empty:
    mlp["task"] = "LUAD driver prediction"
    frames.append(mlp)

resmlp = read_if(os.path.join(LOG_DIR, "ResMLP_LUAD_metrics.csv"))
if not resmlp.empty:
    resmlp["task"] = "LUAD driver prediction"
    frames.append(resmlp)

qc = read_if(os.path.join(LOG_DIR, "QC_model_metrics.csv"))
if not qc.empty:
    qc["task"] = "SeqQC confidence prediction"
    qc["feature_set"] = "QC"
    frames.append(qc)

cross = read_if(os.path.join(LOG_DIR, "coad_gbm_cross_cancer_metrics.csv"))
if not cross.empty:
    frames.append(cross)

final = pd.concat(frames, ignore_index=True)

# Clean misleading rows if any exist
final = final[~((final["model"].astype(str) == "MVF") & (final["feature_set"].astype(str).str.contains("QC", na=False)))]

final = final[["task", "model", "feature_set", "AUROC", "AUPRC", "F1"]]
final.to_csv(os.path.join(LOG_DIR, "BIBM_final_metrics_clean.csv"), index=False)
final.round(4).to_csv(os.path.join(LOG_DIR, "BIBM_final_metrics_clean_rounded.csv"), index=False)

print("Final metrics:")
print(final.round(4))

colors = {
    "RandomForest": "#1f77b4",
    "LightGBM": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "MLP": "#d62728",
    "ResMLP": "#9467bd",
    "MVF": "#8c564b"
}

for task in final["task"].dropna().unique():
    df_task = final[final["task"] == task].copy()
    df_task["label"] = df_task["model"].astype(str) + "\n" + df_task["feature_set"].astype(str)

    for metric in ["AUROC", "AUPRC", "F1"]:
        tmp = df_task.dropna(subset=[metric]).sort_values(metric, ascending=False)
        if tmp.empty:
            continue

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=tmp,
            x="label",
            y=metric,
            hue="model",
            dodge=False,
            palette=colors
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{task}: {metric}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        plt.xticks(rotation=45, ha="right")

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Model", fontsize=8)

        plt.tight_layout()
        fname = f"BIBM_{task.replace(' ', '_').replace('/', '_')}_{metric}.png"
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=300, bbox_inches="tight")
        plt.close()

print("Saved:")
print(os.path.join(LOG_DIR, "BIBM_final_metrics_clean_rounded.csv"))
print("Figures saved:", FIG_DIR)