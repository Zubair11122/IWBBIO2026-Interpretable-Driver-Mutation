import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"
DATA_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/master_mutation_dataset.csv")
FEATURE_FILE = os.path.join(BASE_DIR, "bibm2026_project/data/feature_groups.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "bibm2026_project/outputs")
MODEL_DIR = os.path.join(BASE_DIR, "bibm2026_project/models")
LOG_DIR = os.path.join(BASE_DIR, "bibm2026_project/logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

with open(FEATURE_FILE, "r") as f:
    feature_groups = json.load(f)

features = feature_groups["all_features"]
label_col = feature_groups["label_col"]

X = df[features].values.astype(np.float32)
y = df[label_col].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

class MutationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MutationDataset(X_train, y_train), batch_size=1024, shuffle=True, num_workers=2)
test_loader = DataLoader(MutationDataset(X_test, y_test), batch_size=2048, shuffle=False, num_workers=2)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class ResMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ResMLP(X_train.shape[1]).to(device)

pos = y_train.sum()
neg = len(y_train) - pos
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
print("pos_weight:", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

best_auprc = -1
best_state = None
bad_epochs = 0
patience = 6
epochs = 50

def evaluate():
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.extend(p)
            labels.extend(yb.numpy().reshape(-1))

    probs = np.array(probs)
    labels = np.array(labels)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    pred = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, pred)
    return auroc, auprc, f1, probs, labels

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    auroc, auprc, f1, _, _ = evaluate()

    print(f"Epoch {epoch:02d} | Loss={avg_loss:.5f} | AUROC={auroc:.5f} | AUPRC={auprc:.5f} | F1={f1:.5f}")

    if auprc > best_auprc:
        best_auprc = auprc
        best_state = model.state_dict()
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        print("Early stopping.")
        break

model.load_state_dict(best_state)
auroc, auprc, f1, probs, labels = evaluate()

print("\nFINAL RESMLP RESULT")
print("AUROC:", auroc)
print("AUPRC:", auprc)
print("F1:", f1)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ResMLP_LUAD_all_features.pth"))

pd.DataFrame({
    "y_true": labels,
    "y_proba": probs,
    "y_pred": (probs >= 0.5).astype(int)
}).to_csv(os.path.join(OUTPUT_DIR, "ResMLP_LUAD_all_features_predictions.csv"), index=False)

pd.DataFrame([{
    "model": "ResMLP",
    "feature_set": "functional+signature",
    "AUROC": auroc,
    "AUPRC": auprc,
    "F1": f1
}]).to_csv(os.path.join(LOG_DIR, "ResMLP_LUAD_metrics.csv"), index=False)

print("Saved ResMLP model, predictions, and metrics.")