import os
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"

LUAD_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/master_mutation_dataset.csv")
LUAD_FEATURE_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/feature_groups.json")
QC_FILE = os.path.join(BASE_DIR, "dataset/SeqQC-Former/data_root/features_merged_table.csv")

MODEL_DIR = os.path.join(BASE_DIR, "bibm2026_project/models")
OUTPUT_DIR = os.path.join(BASE_DIR, "bibm2026_project/outputs")
LOG_DIR = os.path.join(BASE_DIR, "bibm2026_project/logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

class Branch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class MVFNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.branches = nn.ModuleList([Branch(d) for d in dims])
        self.head = nn.Sequential(
            nn.Linear(64 * len(dims), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        split_points = np.cumsum([0] + self.dims)
        parts = [x[:, split_points[i]:split_points[i+1]] for i in range(len(self.dims))]
        encoded = [branch(part) for branch, part in zip(self.branches, parts)]
        return self.head(torch.cat(encoded, dim=1))

def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

def predict_batches(model, X, batch_size=4096):
    model.eval()
    probs = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start+batch_size], dtype=torch.float32).to(DEVICE)
            p = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            probs.append(p)

    return np.concatenate(probs)

def train_one(X, y, dims, task, model_name, feature_set, save_prefix):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    model = MVFNet(dims).to(DEVICE)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_auprc = -1
    best_state = None
    bad_epochs = 0
    patience = 5
    epochs = 30
    batch_size = 1024

    print("\n" + "=" * 80)
    print(f"Training {model_name} | {task} | {feature_set}")
    print("Input shape:", X.shape)
    print("View dimensions:", dims)
    print("pos_weight:", pos_weight.item())
    print("=" * 80)

    n = len(X_train)

    for epoch in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(n)
        total_loss = 0.0

        for start in range(0, n, batch_size):
            idx = indices[start:start+batch_size]
            xb = torch.tensor(X_train[idx], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_train[idx], dtype=torch.float32).view(-1, 1).to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / n
        prob = predict_batches(model, X_test)
        m = metrics(y_test, prob)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss={avg_loss:.5f} | "
            f"AUROC={m['AUROC']:.5f} | "
            f"AUPRC={m['AUPRC']:.5f} | "
            f"F1={m['F1']:.5f}"
        )

        if m["AUPRC"] > best_auprc:
            best_auprc = m["AUPRC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    model.to(DEVICE)

    final_prob = predict_batches(model, X_test)
    final_metrics = metrics(y_test, final_prob)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{save_prefix}.pth"))

    pd.DataFrame({
        "y_true": y_test,
        "y_proba": final_prob,
        "y_pred": (final_prob >= 0.5).astype(int)
    }).to_csv(os.path.join(OUTPUT_DIR, f"{save_prefix}_predictions.csv"), index=False)

    result = {
        "task": task,
        "model": model_name,
        "feature_set": feature_set,
        "AUROC": final_metrics["AUROC"],
        "AUPRC": final_metrics["AUPRC"],
        "F1": final_metrics["F1"]
    }

    print("FINAL:", result)
    return result

all_results = []

# -----------------------------
# LUAD module
# -----------------------------
print("\nLoading LUAD dataset...")
luad = pd.read_csv(LUAD_FILE)

with open(LUAD_FEATURE_FILE, "r") as f:
    groups = json.load(f)

functional_features = groups["functional_features"]
signature_features = groups["signature_features"]
label_col = groups["label_col"]

all_luad_features = functional_features + signature_features
luad[all_luad_features] = luad[all_luad_features].apply(pd.to_numeric, errors="coerce").fillna(0)
y_luad = luad[label_col].values

luad_jobs = [
    ("functional", [functional_features]),
    ("signature", [signature_features]),
    ("functional+signature", [functional_features, signature_features]),
]

for feature_set, group_list in luad_jobs:
    X_parts = [luad[g].values.astype(np.float32) for g in group_list]
    X = np.hstack(X_parts)
    dims = [len(g) for g in group_list]

    result = train_one(
        X=X,
        y=y_luad,
        dims=dims,
        task="LUAD driver prediction",
        model_name="MVF",
        feature_set=feature_set,
        save_prefix=f"MVF_LUAD_{feature_set.replace('+', '_plus_')}"
    )

    all_results.append(result)

    pd.DataFrame(all_results).to_csv(os.path.join(LOG_DIR, "MVF_ablation_metrics.csv"), index=False)
    pd.DataFrame(all_results).round(4).to_csv(os.path.join(LOG_DIR, "MVF_ablation_metrics_rounded.csv"), index=False)

# -----------------------------
# SeqQC module: separate dataset
# -----------------------------
print("\nLoading SeqQC dataset...")
qc = pd.read_csv(QC_FILE)

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
    'normal_contamination_flag','homopolymer_length',
    'MAP_missing','RTIM_missing'
]

qc_features = [c for c in qc_features if c in qc.columns]
qc[qc_features] = qc[qc_features].apply(pd.to_numeric, errors="coerce").fillna(0)

X_qc = qc[qc_features].values.astype(np.float32)
y_qc = qc["label"].values

qc_result = train_one(
    X=X_qc,
    y=y_qc,
    dims=[len(qc_features)],
    task="SeqQC confidence prediction",
    model_name="QC-MLP",
    feature_set="QC",
    save_prefix="QC_MLP_SeqQC"
)

all_results.append(qc_result)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(LOG_DIR, "MVF_ablation_metrics.csv"), index=False)
results_df.round(4).to_csv(os.path.join(LOG_DIR, "MVF_ablation_metrics_rounded.csv"), index=False)

print("\nSaved:")
print(os.path.join(LOG_DIR, "MVF_ablation_metrics.csv"))
print(os.path.join(LOG_DIR, "MVF_ablation_metrics_rounded.csv"))
print(results_df.round(4))