import os
import re
import json
import pandas as pd

BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"
INPUT_FILE = os.path.join(BASE_DIR, "dataset/IDEA-LUAD/dataset/mutations_variant_complete.tsv")
OUT_DIR = os.path.join(BASE_DIR, "bibm2026_project/data")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "master_mutation_dataset.csv")
FEATURE_FILE = os.path.join(OUT_DIR, "feature_groups.json")
SUMMARY_FILE = os.path.join(OUT_DIR, "master_dataset_summary.txt")

print("Reading:", INPUT_FILE)
df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
print("Raw shape:", df.shape)

df = df.loc[:, ~df.columns.duplicated()].copy()
print("Shape after duplicate-column removal:", df.shape)

if "is_driver" not in df.columns:
    raise ValueError("Missing column: is_driver")

df["label"] = df["is_driver"].astype(str).str.strip().map({
    "True": 1, "False": 0,
    "true": 1, "false": 0,
    "1": 1, "0": 0
})

df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

print("Label counts:")
print(df["label"].value_counts())

signature_features = [
    c for c in df.columns
    if re.fullmatch(r"sbs[0-9]+[a-z]?", c.lower())
]

meta_cols = [
    c for c in [
        "sample",
        "tumor_sample_barcode",
        "hugo_symbol",
        "chromosome",
        "start_position",
        "reference_allele",
        "tumor_seq_allele2",
        "variant_classification",
        "af",
        "dominant_signature"
    ]
    if c in df.columns
]

exclude_cols = set(meta_cols + [
    "end",
    "amino_acid_change",
    "callers",
    "cancer_type",
    "is_driver",
    "mutation",
    "context",
    "mutation_type",
    "type",
    "label"
])

functional_features = []

for c in df.columns:
    if c in exclude_cols:
        continue
    if c in signature_features:
        continue

    s = pd.to_numeric(df[c], errors="coerce")
    if s.notna().mean() >= 0.20:
        df[c] = s
        functional_features.append(c)

signature_features = list(dict.fromkeys(signature_features))
functional_features = list(dict.fromkeys(functional_features))

all_features = list(dict.fromkeys(signature_features + functional_features))

# Remove any feature also present as metadata
all_features = [c for c in all_features if c not in meta_cols]
signature_features = [c for c in signature_features if c in all_features]
functional_features = [c for c in functional_features if c in all_features]

keep_cols = list(dict.fromkeys(meta_cols + ["label"] + all_features))
master = df[keep_cols].copy()
master = master.loc[:, ~master.columns.duplicated()].copy()

# Convert features safely
valid_features = []
for c in all_features:
    if c not in master.columns:
        continue

    col_obj = master[c]

    # Safety: skip if duplicate selection returns DataFrame
    if isinstance(col_obj, pd.DataFrame):
        print("Skipping duplicated feature:", c)
        continue

    master[c] = pd.to_numeric(col_obj, errors="coerce")
    valid_features.append(c)

all_features = valid_features

# Drop features with >80% missing
missing_rate = master[all_features].isna().mean()
selected_features = missing_rate[missing_rate <= 0.80].index.tolist()

signature_features = [c for c in signature_features if c in selected_features]
functional_features = [c for c in functional_features if c in selected_features]
all_features = list(dict.fromkeys(signature_features + functional_features))

# Median imputation
for c in all_features:
    med = master[c].median()
    if pd.isna(med):
        med = 0
    master[c] = master[c].fillna(med)

final_cols = list(dict.fromkeys(meta_cols + ["label"] + all_features))
master = master[final_cols].copy()
master = master.loc[:, ~master.columns.duplicated()].copy()

print("Final shape:", master.shape)
print("Final label counts:")
print(master["label"].value_counts())
print("Signature features:", len(signature_features))
print("Functional features:", len(functional_features))
print("Total model features:", len(all_features))

master.to_csv(OUT_FILE, index=False)

feature_groups = {
    "signature_features": signature_features,
    "functional_features": functional_features,
    "all_features": all_features,
    "meta_cols": meta_cols,
    "label_col": "label"
}

with open(FEATURE_FILE, "w") as f:
    json.dump(feature_groups, f, indent=2)

with open(SUMMARY_FILE, "w") as f:
    f.write("MASTER DATASET SUMMARY\n")
    f.write("======================\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Final shape: {master.shape}\n")
    f.write("\nLabel counts:\n")
    f.write(str(master["label"].value_counts()))
    f.write("\n\nNumber of signature features: ")
    f.write(str(len(signature_features)))
    f.write("\nNumber of functional features: ")
    f.write(str(len(functional_features)))
    f.write("\nTotal features: ")
    f.write(str(len(all_features)))
    f.write("\n\nSignature features:\n")
    f.write("\n".join(signature_features))
    f.write("\n\nFunctional features:\n")
    f.write("\n".join(functional_features))

print("Saved:", OUT_FILE)
print("Saved:", FEATURE_FILE)
print("Saved:", SUMMARY_FILE)