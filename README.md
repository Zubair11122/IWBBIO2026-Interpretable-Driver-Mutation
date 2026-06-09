# IWBBIO2026-Interpretable-Driver-Mutation

This repository contains code, processed outputs, evaluation metrics, and figures for the conference paper:

**Interpretable Deep Learning for Driver Mutation Prediction Using Functional and Mutational Signature Features Across Multiple Cancer Types**

## 1. Project Overview

Accurate identification of cancer driver mutations is essential for precision oncology, but remains challenging because of abundant passenger mutations, heterogeneous genomic landscapes, and sequencing artifacts. This project presents a unified interpretable deep learning framework that integrates functional pathogenicity annotations, mutational signature features, and sequencing quality-control information.

The framework contains three complementary modules:

1. **LUAD driver mutation prediction**
2. **SeqQC sequencing-confidence prediction**
3. **COAD/GBM cross-cancer validation**

Model interpretation is performed using SHAP-based feature-attribution analysis.

## 2. Repository Structure

```text
IWBBIO2026-Interpretable-Driver-Mutation/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── data_description/
├── preprocessing/
├── models/
├── experiments/
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── shap_values/
├── figures/
├── shap_analysis/
└── scripts/
```

## 3. Dataset Description

This study uses three types of datasets:

### LUAD driver mutation dataset

The LUAD module contains 200,432 somatic mutations represented by 229 predictive features, including 143 functional genomic descriptors and 86 SBS mutational signature features.

### SeqQC confidence prediction dataset

The SeqQC module contains 89,447 candidate variants represented by 35 sequencing quality-control features. This module evaluates variant reliability independently from biological driver status.

### COAD/GBM cross-cancer validation dataset

The cross-cancer module contains 1,232,949 somatic mutations from colorectal adenocarcinoma and glioblastoma cohorts. This module evaluates transferability between biologically distinct cancer types.

Raw controlled-access datasets are not included in this repository. Public datasets and access-controlled resources are described in the paper.

## 4. Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/IWBBIO2026-Interpretable-Driver-Mutation.git
cd IWBBIO2026-Interpretable-Driver-Mutation
```

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate iwbbio-driver
```

Alternatively, install dependencies using pip:

```bash
pip install -r requirements.txt
```

## 5. Running LUAD Driver Mutation Prediction

```bash
python scripts/run_luad_prediction.py \
  --config experiments/luad_config.yaml \
  --output results/predictions/
```

Expected outputs include LUAD prediction files and performance metrics for Random Forest, LightGBM, XGBoost, MLP, and ResMLP models.

## 6. Running SeqQC Confidence Prediction

```bash
python scripts/run_seqqc_prediction.py \
  --config experiments/seqqc_config.yaml \
  --output results/predictions/
```

Expected outputs include SeqQC prediction files and metrics for baseline QC models and QC-MLP.

## 7. Running COAD/GBM Cross-Cancer Validation

```bash
python scripts/run_cross_cancer_validation.py \
  --config experiments/cross_cancer_config.yaml \
  --output results/predictions/
```

This experiment trains models on one cancer type and evaluates them on the other:

* GBM to COAD
* COAD to GBM

## 8. Reproducing Tables and Figures

To reproduce final metrics tables:

```bash
python scripts/make_tables.py \
  --metrics_dir results/metrics/ \
  --output results/
```

To reproduce figures:

```bash
python scripts/make_figures.py \
  --metrics_dir results/metrics/ \
  --shap_dir results/shap_values/ \
  --output figures/
```

The main figures include:

* LUAD driver prediction performance
* SeqQC confidence prediction performance
* COAD/GBM cross-cancer validation performance
* SHAP feature-importance summaries

## 9. SHAP Analysis

SHAP feature-importance files are stored in:

```text
results/shap_values/
```

To regenerate SHAP summaries:

```bash
python shap_analysis/run_shap_analysis.py \
  --config experiments/shap_config.yaml \
  --output results/shap_values/
```

## 10. Results

Key result files are stored in:

```text
results/metrics/
results/predictions/
results/shap_values/
figures/
```

Main reported results include:

* LUAD ResMLP: AUROC = 0.9779, AUPRC = 0.9420, F1-score = 0.8615
* SeqQC QC-MLP: AUROC = 0.6348, AUPRC = 0.6239
* COAD-to-GBM ResMLP transfer: AUROC = 0.6824, AUPRC = 0.3770
* GBM-to-COAD ResMLP transfer: AUROC = 0.5941, AUPRC = 0.2661

## 11. Citation

If you use this repository, please cite:

```text
Zubair, M., Hui, Q., and Li, J. Interpretable Deep Learning for Driver Mutation Prediction Using Functional and Mutational Signature Features Across Multiple Cancer Types. IWBBIO 2026.
```

Additional related work:

```text
Zubair, M., Li, J., and Qian, J. Signature-aware deep learning reveals distinct driver gene programs and mutational processes in glioblastoma and colon adenocarcinoma. Computational Biology and Chemistry, 123, 108984, 2026.
```

## 12. License

This repository is released for academic and research use. See the LICENSE file for details.

## 13. Contact

For questions, please contact:

**Muhammad Zubair**
College of Computer Science, Beijing University of Technology
ORCID: <0009-0004-2636-255X>
