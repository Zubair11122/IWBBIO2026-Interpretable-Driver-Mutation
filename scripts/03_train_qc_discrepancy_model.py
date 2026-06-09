import os
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
QC_FILE = os.path.join(BASE_DIR, "dataset/SeqQC-Former/data_root/features_merged_table.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "bibm2026_project/outputs")
MODEL_DIR = os.path.join(BASE_DIR, "bibm2026_project/models")
LOG_DIR = os.path.join(BASE_DIR, "bibm2026_project/logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(QC_FILE)

# QC feature columns
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

# Target label
y_col = 'label'

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df[qc_features], df[y_col], test_size=0.2, stratify=df[y_col], random_state=42
)

# -----------------------------
# Train models
# -----------------------------
results = []

for model_name in ['RandomForest','LightGBM','XGBoost']:
    print(f"\n=== Training {model_name} on QC features ===")
    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=500, max_depth=None,
            n_jobs=-1, class_weight='balanced', random_state=42
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)
        joblib.dump(model, os.path.join(MODEL_DIR,f"RF_QC.joblib"))

    elif model_name == 'LightGBM':
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
        lgb_params = {
            "objective":"binary","metric":"binary_logloss",
            "verbosity":-1,"boosting_type":"gbdt",
            "n_jobs":-1,"seed":42,
            "scale_pos_weight":(len(y_train)-sum(y_train))/sum(y_train)
        }
        model = lgb.train(
            lgb_params, lgb_train, num_boost_round=500,
            valid_sets=[lgb_train,lgb_eval], verbose_eval=False
        )
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba>=0.5).astype(int)
        model.save_model(os.path.join(MODEL_DIR,f"LGBM_QC.txt"))

    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            n_jobs=-1, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)
        joblib.dump(model, os.path.join(MODEL_DIR,f"XGB_QC.joblib"))

    # -----------------------------
    # Evaluate
    # -----------------------------
    metrics = {
        "model": model_name,
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "AUPRC": average_precision_score(y_test, y_pred_proba),
        "F1": f1_score(y_test, y_pred)
    }
    results.append(metrics)
    pd.DataFrame({"y_true":y_test,"y_pred":y_pred,"y_proba":y_pred_proba}).to_csv(
        os.path.join(OUTPUT_DIR,f"{model_name}_QC_predictions.csv"), index=False
    )
    print(f"{model_name} done: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, F1={metrics['F1']:.4f}")

# -----------------------------
# Save metrics
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(LOG_DIR,"QC_model_metrics.csv"), index=False)
print("\nQC models trained. Metrics saved to QC_model_metrics.csv")