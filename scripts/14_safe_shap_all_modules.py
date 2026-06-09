import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"

MODEL_DIR = os.path.join(BASE_DIR, "bibm2026_project/models")
FIG_DIR = os.path.join(BASE_DIR, "bibm2026_project/figures")
LOG_DIR = os.path.join(BASE_DIR, "bibm2026_project/logs")
DATA_DIR = os.path.join(BASE_DIR, "bibm2026_project/data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def positive_class_shap(values):
    if isinstance(values, list):
        return values[1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr

def get_model_for_shap(model):
    # If sklearn Pipeline, use final estimator when possible
    if hasattr(model, "named_steps"):
        if "rf" in model.named_steps:
            return model.named_steps["rf"], model.named_steps.get("scaler", None)
    return model, None

def run_tree_shap(model, X, feature_names, prefix):
    print("\nRunning SHAP:", prefix, X.shape)

    model_for_shap, scaler = get_model_for_shap(model)

    X_local = X.copy()
    if scaler is not None:
        X_values = scaler.transform(X_local.values)
        X_local = pd.DataFrame(X_values, columns=feature_names)

    explainer = shap.TreeExplainer(model_for_shap)
    values = explainer.shap_values(X_local)
    values = positive_class_shap(values)

    if values.shape[1] != len(feature_names):
        raise ValueError(f"SHAP shape mismatch: {values.shape} vs {len(feature_names)}")

    top = (
        pd.DataFrame(np.abs(values), columns=feature_names)
        .mean()
        .sort_values(ascending=False)
    )

    top.to_csv(os.path.join(LOG_DIR, f"{prefix}_mean_abs_shap_all.csv"))
    top.head(30).to_csv(os.path.join(LOG_DIR, f"{prefix}_top30_features.csv"))

    plt.figure(figsize=(9, 7))
    top.head(20).sort_values().plot(kind="barh")
    plt.xlabel("Mean absolute SHAP value")
    plt.title(prefix.replace("_", " "))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{prefix}_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(values, X_local, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{prefix}_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", prefix)

# LUAD RF functional
luad_file = os.path.join(DATA_DIR, "master_mutation_dataset.csv")
luad_group_file = os.path.join(DATA_DIR, "feature_groups.json")

if os.path.exists(luad_file) and os.path.exists(os.path.join(MODEL_DIR, "RF_functional.joblib")):
    luad = pd.read_csv(luad_file)
    with open(luad_group_file, "r") as f:
        groups = json.load(f)

    feats = groups["functional_features"]
    X = luad[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.sample(n=min(1000, len(X)), random_state=42)

    model = joblib.load(os.path.join(MODEL_DIR, "RF_functional.joblib"))
    run_tree_shap(model, X, feats, "SHAP_LUAD_RF_functional")
else:
    print("Skipping LUAD SHAP: missing data or RF_functional model")

# SeqQC RF
qc_file = os.path.join(BASE_DIR, "dataset/SeqQC-Former/data_root/features_merged_table.csv")

if os.path.exists(qc_file) and os.path.exists(os.path.join(MODEL_DIR, "RF_QC.joblib")):
    qc = pd.read_csv(qc_file)

    qc_features = [
        'DP','AD','VAF','MQ','SB','MAP','RTIM',
        'tumor_depth','tumor_alt_count','tumor_alt_fraction',
        'tumor_mapq_mean','tumor_baseq_mean',
        'tumor_strand_bias','tumor_orientation_bias',
        'tumor_clipped_fraction','tumor_mismatch_fraction',
        'tumor_indel_fraction','tumor_read_position_bias',
        'normal_depth','normal_alt_count','normal_alt_fraction',
        'normal_mapq_mean','normal_baseq_mean',
        'normal_strand_bias','normal_orientation_bias',
        'normal_clipped_fraction','normal_mismatch_fraction',
        'normal_indel_fraction','normal_read_position_bias',
        'delta_alt_fraction','germline_support_flag',
        'normal_contamination_flag','homopolymer_length'
    ]

    qc_features = [c for c in qc_features if c in qc.columns]
    X = qc[qc_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.sample(n=min(1000, len(X)), random_state=42)

    model = joblib.load(os.path.join(MODEL_DIR, "RF_QC.joblib"))
    run_tree_shap(model, X, qc_features, "SHAP_SeqQC_RF_QC")
else:
    print("Skipping SeqQC SHAP: missing data or RF_QC model")

# COAD/GBM RF cross-cancer
cg_file = os.path.join(DATA_DIR, "coad_gbm_master_dataset.csv")
cg_group_file = os.path.join(DATA_DIR, "coad_gbm_feature_groups.json")

if os.path.exists(cg_file) and os.path.exists(cg_group_file):
    cg = pd.read_csv(cg_file)
    with open(cg_group_file, "r") as f:
        cg_groups = json.load(f)

    feats = cg_groups["all_features"]
    rf_models = [m for m in os.listdir(MODEL_DIR) if m.startswith("RF_cross_") and m.endswith(".joblib")]

    if rf_models:
        model_name = sorted(rf_models)[0]
        model = joblib.load(os.path.join(MODEL_DIR, model_name))

        X = cg[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
        X = X.sample(n=min(1000, len(X)), random_state=42)

        run_tree_shap(model, X, feats, "SHAP_COAD_GBM_RF_cross_reference")
    else:
        print("Skipping COAD/GBM SHAP: no RF_cross model found")
else:
    print("Skipping COAD/GBM SHAP: missing dataset or feature file")

print("All safe SHAP completed.")