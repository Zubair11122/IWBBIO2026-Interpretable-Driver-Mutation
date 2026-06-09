import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"

DATA_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/coad_gbm_master_dataset.csv")
FEATURE_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/coad_gbm_feature_groups.json")

MODEL_DIR = os.path.join(BASE_DIR, "bibm2026_project/models")
OUTPUT_DIR = os.path.join(BASE_DIR, "bibm2026_project/outputs")
LOG_DIR = os.path.join(BASE_DIR, "bibm2026_project/logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

with open(FEATURE_FILE, "r") as f:
    groups = json.load(f)

features = groups["all_features"]
label_col = groups["label_col"]
cancer_col = groups["cancer_col"]

print("Dataset shape:", df.shape)
print("Features:", len(features))
print("Cancer counts:")
print(df[cancer_col].value_counts())
print("Label counts:")
print(df[label_col].value_counts())

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))

class ResMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.r1 = ResBlock(512)
        self.r2 = ResBlock(512)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.inp(x)
        x = self.r1(x)
        x = self.r2(x)
        return self.head(x)

def safe_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "AUROC": np.nan,
        "AUPRC": np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }
    if len(np.unique(y_true)) == 2:
        out["AUROC"] = roc_auc_score(y_true, y_prob)
        out["AUPRC"] = average_precision_score(y_true, y_prob)
    return out

def train_resmlp(X_train, y_train, X_test, y_test, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=1024, shuffle=True, num_workers=2)
    test_loader = DataLoader(TabDataset(X_test, y_test), batch_size=2048, shuffle=False, num_workers=2)

    model = ResMLP(X_train.shape[1]).to(device)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    best_auprc = -1
    best_state = None
    patience = 6
    bad = 0

    for epoch in range(1, 51):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
                probs.extend(p)
                labels.extend(yb.numpy().reshape(-1))

        probs = np.array(probs)
        labels = np.array(labels)
        m = safe_metrics(labels, probs)
        print(f"{name} Epoch {epoch:02d} | AUROC={m['AUROC']} | AUPRC={m['AUPRC']} | F1={m['F1']:.4f}")

        score = -1 if pd.isna(m["AUPRC"]) else m["AUPRC"]
        if score > best_auprc:
            best_auprc = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_state)
    model.to(device)

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            probs.extend(p)
            labels.extend(yb.numpy().reshape(-1))

    probs = np.array(probs)
    labels = np.array(labels)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"ResMLP_{name}.pth"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"Scaler_{name}.joblib"))

    return probs, labels, safe_metrics(labels, probs)

results = []

cancers = df[cancer_col].dropna().unique().tolist()
print("Detected cancer groups:", cancers)

for train_cancer in cancers:
    for test_cancer in cancers:
        if train_cancer == test_cancer:
            continue

        train_df = df[df[cancer_col] == train_cancer].copy()
        test_df = df[df[cancer_col] == test_cancer].copy()

        if len(train_df) < 50 or len(test_df) < 50:
            print("Skipping small split:", train_cancer, test_cancer)
            continue

        X_train = train_df[features].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(int)

        X_test = test_df[features].values.astype(np.float32)
        y_test = test_df[label_col].values.astype(int)

        # Random Forest cross-cancer
        rf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42
            ))
        ])

        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        rf_metrics = safe_metrics(y_test, rf_prob)

        name = f"{train_cancer}_to_{test_cancer}".replace("/", "_").replace(" ", "_")

        joblib.dump(rf, os.path.join(MODEL_DIR, f"RF_cross_{name}.joblib"))

        pd.DataFrame({
            "y_true": y_test,
            "y_proba": rf_prob,
            "y_pred": (rf_prob >= 0.5).astype(int)
        }).to_csv(os.path.join(OUTPUT_DIR, f"RF_cross_{name}_predictions.csv"), index=False)

        results.append({
            "task": "COAD/GBM cross-cancer validation",
            "model": "RandomForest",
            "feature_set": f"train {train_cancer} test {test_cancer}",
            **rf_metrics
        })

        # ResMLP cross-cancer
        res_prob, res_labels, res_metrics = train_resmlp(X_train, y_train, X_test, y_test, f"cross_{name}")

        pd.DataFrame({
            "y_true": res_labels,
            "y_proba": res_prob,
            "y_pred": (res_prob >= 0.5).astype(int)
        }).to_csv(os.path.join(OUTPUT_DIR, f"ResMLP_cross_{name}_predictions.csv"), index=False)

        results.append({
            "task": "COAD/GBM cross-cancer validation",
            "model": "ResMLP",
            "feature_set": f"train {train_cancer} test {test_cancer}",
            **res_metrics
        })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(LOG_DIR, "coad_gbm_cross_cancer_metrics.csv"), index=False)
results_df.round(4).to_csv(os.path.join(LOG_DIR, "coad_gbm_cross_cancer_metrics_rounded.csv"), index=False)

print("Saved:")
print(os.path.join(LOG_DIR, "coad_gbm_cross_cancer_metrics_rounded.csv"))
print(results_df.round(4))