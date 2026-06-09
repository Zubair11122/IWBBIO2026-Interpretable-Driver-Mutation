import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
import xgboost as xgb
import joblib

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"
MASTER_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/master_mutation_dataset.csv")
QC_FILE = os.path.join(BASE_DIR, "dataset/SeqQC-Former/data_root/features_merged_table.csv")
FEATURE_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/feature_groups.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "bibm2026_project/outputs")
MODEL_DIR = os.path.join(BASE_DIR, "bibm2026_project/models")
LOG_DIR = os.path.join(BASE_DIR, "bibm2026_project/logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
master_df = pd.read_csv(MASTER_FILE)
qc_df = pd.read_csv(QC_FILE)

with open(FEATURE_FILE, "r") as f:
    feature_groups = json.load(f)

all_features = feature_groups["all_features"]
meta_cols = feature_groups["meta_cols"]
label_col = feature_groups["label_col"]

# -----------------------------
# Merge QC features
# -----------------------------
# Identify common columns for merge: sample + mutation coordinates
# Adjust if your SeqQC file has different column names
merge_cols = ["sample", "chromosome", "start_position", "reference_allele", "tumor_seq_allele2"]
for col in merge_cols:
    if col not in master_df.columns or col not in qc_df.columns:
        raise ValueError(f"Merge column missing: {col}")

# Merge QC features
qc_features = [c for c in qc_df.columns if c not in merge_cols]
merged_df = master_df.merge(qc_df[merge_cols + qc_features], on=merge_cols, how="left")
print(f"Shape after merging QC features: {merged_df.shape}")

# Impute missing QC features with median
for c in qc_features:
    if c in merged_df.columns:
        med = merged_df[c].median()
        merged_df[c] = merged_df[c].fillna(med)

# -----------------------------
# Final feature sets
# -----------------------------
proposed_features = all_features + qc_features

# -----------------------------
# Split train/test
# -----------------------------
X = merged_df[proposed_features]
y = merged_df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Train models
# -----------------------------
feature_sets = {
    "functional+QC": proposed_features
}

results = []

for name, feats in feature_sets.items():
    print(f"\n=== Training proposed model: {name} ({len(feats)} features) ===")
    X_tr, X_te = X_train[feats], X_test[feats]

    # Random Forest
    print("Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_tr, y_train)
    y_pred_proba = rf.predict_proba(X_te)[:,1]
    y_pred = rf.predict(X_te)
    rf_metrics = {
        "model": "RandomForest",
        "feature_set": name,
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "AUPRC": average_precision_score(y_test, y_pred_proba),
        "F1": f1_score(y_test, y_pred)
    }
    results.append(rf_metrics)
    joblib.dump(rf, os.path.join(MODEL_DIR, f"RF_{name}.joblib"))
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_pred_proba}).to_csv(
        os.path.join(OUTPUT_DIR, f"RF_{name}_predictions.csv"), index=False
    )
    print("Random Forest done.")

    # LightGBM
    print("LightGBM...")
    lgb_train = lgb.Dataset(X_tr, label=y_train)
    lgb_eval = lgb.Dataset(X_te, label=y_test, reference=lgb_train)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "seed": 42,
        "scale_pos_weight": (len(y_train) - sum(y_train)) / sum(y_train)
    }
    lgbm = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=False
    )
    y_pred_proba = lgbm.predict(X_te)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    lgb_metrics = {
        "model": "LightGBM",
        "feature_set": name,
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "AUPRC": average_precision_score(y_test, y_pred_proba),
        "F1": f1_score(y_test, y_pred)
    }
    results.append(lgb_metrics)
    lgbm.save_model(os.path.join(MODEL_DIR, f"LGBM_{name}.txt"))
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_pred_proba}).to_csv(
        os.path.join(OUTPUT_DIR, f"LGBM_{name}_predictions.csv"), index=False
    )
    print("LightGBM done.")

    # XGBoost
    print("XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42
    )
    xgb_model.fit(X_tr, y_train)
    y_pred_proba = xgb_model.predict_proba(X_te)[:,1]
    y_pred = xgb_model.predict(X_te)
    xgb_metrics = {
        "model": "XGBoost",
        "feature_set": name,
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "AUPRC": average_precision_score(y_test, y_pred_proba),
        "F1": f1_score(y_test, y_pred)
    }
    results.append(xgb_metrics)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, f"XGB_{name}.joblib"))
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_pred_proba}).to_csv(
        os.path.join(OUTPUT_DIR, f"XGB_{name}_predictions.csv"), index=False
    )
    print("XGBoost done.")

# -----------------------------
# Save results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(LOG_DIR, "proposed_model_metrics.csv"), index=False)
print("\nProposed model training completed. Metrics saved to proposed_model_metrics.csv")