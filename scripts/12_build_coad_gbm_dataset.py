import os
import json
import pandas as pd
import numpy as np

BASE_DIR = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/DAAI-Platform"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset/ResMLP-GL/Dataset/mutations_with_clinical_combined.tsv"
)

OUT_DIR = os.path.join(BASE_DIR, "bibm2026_project/data")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "coad_gbm_master_dataset.csv")
FEATURE_FILE = os.path.join(OUT_DIR, "coad_gbm_feature_groups.json")
SUMMARY_FILE = os.path.join(OUT_DIR, "coad_gbm_dataset_summary.txt")

print("Reading:", INPUT_FILE)
df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

df = df.loc[:, ~df.columns.duplicated()].copy()

# -----------------------------
# Detect cancer column
# -----------------------------
cancer_candidates = [
    "Cancer_Type", "cancer_type", "Tumor_Type", "tumor_type",
    "project", "cohort", "dataset", "disease", "Disease"
]

cancer_col = None
for c in cancer_candidates:
    if c in df.columns:
        cancer_col = c
        break

if cancer_col is None:
    raise ValueError("Could not find cancer type column.")

print("Cancer column:", cancer_col)
print(df[cancer_col].value_counts(dropna=False).head(20))

# -----------------------------
# Detect label column
# -----------------------------
label_candidates = [
    "Is_Driver", "is_driver", "label", "driver_label",
    "driver", "is_cancer_driver", "Driver",
    "is_driver_gene", "driver_status"
]

label_col = None
for c in label_candidates:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError("Could not find driver label column.")

print("Label column:", label_col)

def map_label(x):
    s = str(x).strip().lower()

    if s in ["1", "true", "yes", "driver", "positive", "pathogenic", "likely_pathogenic"]:
        return 1

    if s in ["0", "false", "no", "passenger", "negative", "benign", "unknown"]:
        return 0

    try:
        v = float(s)
        if v == 1:
            return 1
        if v == 0:
            return 0
    except:
        pass

    return np.nan

df["label"] = df[label_col].apply(map_label)
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

print("After label cleaning:", df.shape)
print("Label counts:")
print(df["label"].value_counts())

# -----------------------------
# Normalize cancer names
# -----------------------------
df["cancer_type_clean"] = df[cancer_col].astype(str).str.upper().str.strip()

mask = df["cancer_type_clean"].str.contains("GBM|COAD|COLON|GLIO", regex=True, na=False)
if mask.sum() > 0:
    df = df[mask].copy()

print("Cancer counts after filtering:")
print(df["cancer_type_clean"].value_counts())

# -----------------------------
# Signature features
# -----------------------------
signature_features = [
    c for c in df.columns
    if c.upper().startswith("SBS")
]

# -----------------------------
# Metadata / excluded columns
# -----------------------------
meta_like = set([
    cancer_col, label_col, "label", "cancer_type_clean",
    "sample", "Sample", "sample_id", "Sample ID",
    "Tumor_Sample_Barcode", "tumor_sample_barcode",
    "patient", "patient_id", "case_id",
    "gene", "Gene", "Hugo_Symbol", "hugo_symbol",
    "chrom", "Chromosome", "chromosome",
    "pos", "position", "Start_Position", "start_position",
    "end", "End_Position", "end_position",
    "ref", "Reference_Allele", "reference_allele",
    "alt", "Tumor_Seq_Allele2", "tumor_seq_allele2",
    "Variant_Classification", "variant_classification",
    "Amino_Acid_Change",
    "Mutation", "Context", "Mutation_Type", "Type",
    "callers", "Dominant_Signature"
])

# -----------------------------
# Numeric features
# -----------------------------
numeric_features = []

for c in df.columns:
    if c in meta_like:
        continue
    if c in signature_features:
        continue

    converted = pd.to_numeric(df[c], errors="coerce")
    numeric_rate = converted.notna().mean()

    if numeric_rate >= 0.50:
        df[c] = converted
        numeric_features.append(c)

# Keep AF, DP, survival months if available
important_numeric = ["AF", "DP", "Overall Survival (Months)"]
for c in important_numeric:
    if c in df.columns and c not in numeric_features:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().mean() >= 0.20:
            df[c] = converted
            numeric_features.append(c)

# -----------------------------
# Clinical categorical features
# -----------------------------
clinical_categorical_candidates = [
    "Overall Survival Status"
]

categorical_features = [
    c for c in clinical_categorical_candidates
    if c in df.columns
]

encoded_feature_names = []
if categorical_features:
    encoded_df = pd.get_dummies(
        df[categorical_features].astype(str).fillna("missing"),
        prefix=categorical_features,
        dummy_na=False
    )
    encoded_feature_names = encoded_df.columns.tolist()
    df = pd.concat([df, encoded_df], axis=1)

clinical_encoded_features = encoded_feature_names

# -----------------------------
# Build feature groups
# -----------------------------
signature_features = [c for c in signature_features if c in df.columns]

all_features = list(dict.fromkeys(
    signature_features + numeric_features + clinical_encoded_features
))

valid_features = []

for c in all_features:
    s = pd.to_numeric(df[c], errors="coerce")
    if s.notna().mean() >= 0.20:
        df[c] = s
        valid_features.append(c)

all_features = valid_features
signature_features = [c for c in signature_features if c in all_features]
numeric_features = [c for c in numeric_features if c in all_features]
clinical_encoded_features = [c for c in clinical_encoded_features if c in all_features]

# Median imputation
for c in all_features:
    med = df[c].median()
    if pd.isna(med):
        med = 0
    df[c] = df[c].fillna(med)

meta_cols = [
    c for c in [
        cancer_col,
        "cancer_type_clean",
        "sample",
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Variant_Classification",
        "case_id"
    ]
    if c in df.columns
]

keep_cols = list(dict.fromkeys(meta_cols + ["label"] + all_features))
out = df[keep_cols].copy()

print("Final shape:", out.shape)
print("Feature count:", len(all_features))
print("Signature features:", len(signature_features))
print("Numeric features:", len(numeric_features))
print("Clinical encoded features:", len(clinical_encoded_features))

print("Final cancer counts:")
print(out["cancer_type_clean"].value_counts())

print("Final label counts:")
print(out["label"].value_counts())

out.to_csv(OUT_FILE, index=False)

feature_groups = {
    "signature_features": signature_features,
    "numeric_features": numeric_features,
    "clinical_encoded_features": clinical_encoded_features,
    "all_features": all_features,
    "meta_cols": meta_cols,
    "label_col": "label",
    "cancer_col": "cancer_type_clean"
}

with open(FEATURE_FILE, "w") as f:
    json.dump(feature_groups, f, indent=2)

with open(SUMMARY_FILE, "w") as f:
    f.write("COAD/GBM DATASET SUMMARY\n")
    f.write("========================\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Final shape: {out.shape}\n")

    f.write("\nCancer counts:\n")
    f.write(str(out["cancer_type_clean"].value_counts()))

    f.write("\n\nLabel counts:\n")
    f.write(str(out["label"].value_counts()))

    f.write(f"\n\nTotal features: {len(all_features)}\n")
    f.write(f"Signature features: {len(signature_features)}\n")
    f.write(f"Numeric features: {len(numeric_features)}\n")
    f.write(f"Clinical encoded features: {len(clinical_encoded_features)}\n")

    f.write("\n\nSignature features:\n")
    f.write("\n".join(signature_features))

    f.write("\n\nNumeric features:\n")
    f.write("\n".join(numeric_features))

    f.write("\n\nClinical encoded features:\n")
    f.write("\n".join(clinical_encoded_features))

print("Saved:", OUT_FILE)
print("Saved:", FEATURE_FILE)
print("Saved:", SUMMARY_FILE)