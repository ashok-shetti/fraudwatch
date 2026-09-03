# 🛡️ FraudWatch: Real-Time Fraud Risk Intelligence Engine
> **End-to-End Machine Learning System for Large-Scale Credit Card Fraud Detection on the IEEE-CIS Dataset**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v3.2.0-EB4F27?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-v4.6.0-2E8B57)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

**FraudWatch** is an enterprise-grade, real-time fraud detection pipeline trained on the benchmark [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) (590,540 transactions across 394 features). The system addresses severe class imbalance, high-cardinality categorical entities, and complex identity networks using gradient boosted decision trees (XGBoost & LightGBM).

The production model achieves an **Out-Of-Fold (OOF) ROC-AUC of 0.9694** with **90.0% precision** on fraud transactions at a 0.5 decision threshold, preventing high false-alarm rates in high-throughput transaction environments. The pipeline is packaged into a containerizable **Streamlit** dashboard supporting single-transaction risk scoring (<5 ms inference) and bulk CSV batch processing.

---

## 💼 Business Problem & Engineering Challenges

| Challenge | Impact on Business | Engineering Solution |
| :--- | :--- | :--- |
| **Severe Class Imbalance** | Fraud represents only **3.5%** of transactions (27.58:1 ratio). Naive classifiers minimize loss by predicting majority class. | Calibrated cost-sensitive loss weighting (`scale_pos_weight = 27.58`) and 5-Fold Stratified Cross-Validation. |
| **High-Cardinality Categoricals** | Categoricals like `P_emaildomain`, `DeviceInfo`, and card identifiers contain thousands of rare levels prone to overfitting. | **Smoothed Target Encoding** with global empirical prior ($\alpha=50$) for high-predictive variables, paired with **Frequency Encoding**. |
| **Extreme Sparsity & High Dimensionality** | 390+ raw features with block-missing structures in V-features (engineered identity signals). | Automated missingness profiling dropping features with >90% nulls, combined with vectorized numeric constant imputation (`-999`). |
| **Inference Latency vs. Precision** | False positives degrade customer trust; false negatives lead to direct capital loss. | Selected **XGBoost Histogram (`tree_method='hist'`)** achieving **90% precision** and **0.80 F1-score**, optimized for fast memory mapping. |

---

## 🏗️ End-to-End Pipeline Architecture

```
[ Raw Transaction & Identity Data ] 
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ 01. Preprocessing & Data Cleaning                     │
│  - Merging Transaction & Identity logs                 │
│  - Dropping columns with >90% null values (74 cols)    │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ 02. Exploratory Data Analysis & Anomaly Profiling      │
│  - Temporal distribution analysis (TransactionDT)      │
│  - High-risk domain & card routing pattern analysis    │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ 03. Feature Engineering & Artifact Serialization       │
│  - Cyclic time feature extraction (Hour, Weekday)      │
│  - Log-transformed monetary exposure (TransactionAmt)  │
│  - Bayesian Smoothed Target Encoding (α=50)            │
│  - Serialization: preprocessing_bundle.pkl             │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ 04 & 05. Model Training & 5-Fold Stratified CV         │
│  - LightGBM Baseline (OOF AUC: 0.9640)                 │
│  - XGBoost Hist GBDT (OOF AUC: 0.9694) ──★ Champion    │
│  - Serialization: xgb_final_model.pkl                  │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Production Deployment (Streamlit Real-Time Engine)     │
│  - Single Transaction Risk Scoring (<5ms latency)      │
│  - Bulk Batch CSV Scoring with dynamic thresholding    │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Model Evaluation & Benchmarks

Models were evaluated across 5 stratified folds on the full 590,540 training records to prevent data leakage and out-of-distribution bias.

### Overall Validation Performance

| Metric | LightGBM Baseline | XGBoost Final (Champion) | Relative Lift |
| :--- | :---: | :---: | :---: |
| **OOF ROC-AUC** | `0.964013` | **`0.969376`** | **+0.0054** |
| **Fraud Precision (Class 1)** | `0.69` (69%) | **`0.90` (90%)** | **+21.0%** |
| **Fraud Recall (Class 1)** | **`0.75` (75%)** | `0.72` (72%) | -3.0% |
| **Fraud F1-Score** | `0.72` | **`0.80`** | **+0.08** |
| **Overall Accuracy** | `0.98` | **`0.99`** | **+1.0%** |
| **Training Latency (5 Folds)** | **~7 min 03 s** | ~14 min 32 s | — |

### Stratified 5-Fold Validation Breakdown

| Fold | LightGBM AUC | XGBoost AUC |
| :---: | :---: | :---: |
| **Fold 1** | 0.962793 | **0.967716** |
| **Fold 2** | 0.964347 | **0.968669** |
| **Fold 3** | 0.963601 | **0.968558** |
| **Fold 4** | 0.964751 | **0.971080** |
| **Fold 5** | 0.965737 | **0.970959** |
| **Mean ± Std** | `0.9640 ± 0.0010` | **`0.9694 ± 0.0014`** |

### Top Predictive Features (Feature Importance)
- **Identity & Behavioral Anomaly Vectors**: `V258` (19.6%), `V70` (12.3%), `V294` (3.7%), `V91` (3.7%), `V201` (3.3%)
- **Transaction & Routing Attributes**: `card1`, `card2`, `TransactionAmt_log`, `addr1`, `addr2`
- **Categorical Risk Encodings**: `P_emaildomain_target_enc`, `ProductCD_target_enc`, `DeviceType_target_enc`

---

## 📁 Repository Structure

```text
FraudWatch/
├── data/                               # Dataset directory (raw and processed data)
│   ├── info.txt                        # Dataset schema & description
│   └── test_transaction.csv            # Sample test transactions for validation
├── models/                             # Production serialized model artifacts
│   ├── preprocessing_bundle.pkl        # Encodings, imputers, & feature metadata
│   ├── lgbm_baseline_model.pkl         # Trained LightGBM model artifact
│   └── xgb_final_model.pkl             # Trained Champion XGBoost model artifact
├── notebooks/                          # Modular Jupyter experimentation workflows
│   ├── 01_data_preprocessing.ipynb     # Data ingestion, merging, and null cleaning
│   ├── 02_EDA.ipynb                    # Imbalance profiling, bivariate & time analysis
│   ├── 03_feature_engineering.ipynb    # Target encoding, frequency encoding, imputations
│   ├── 04_model_training_lightgbm.ipynb# 5-fold CV LightGBM baseline
│   └── 05_model_training_xgboost.ipynb # 5-fold CV XGBoost with hyperparameter tuning
├── app.py                              # Production Streamlit web application
├── requirements.txt                    # Pinned Python package dependencies
├── .gitignore                          # Standard git ignore rules
└── README.md                           # Technical documentation & project guide
```

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10 or 3.11 installed
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/FraudWatch.git
cd FraudWatch
```

### 3. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Application Capabilities

- **🔍 Single Transaction Scoring**:
  - Live parameter input: Transaction Amount, Product Code, Card Issuer, Purchaser Email Domain, Device Type, and Time of Day.
  - One-click **Quick-Fill Presets** (*Low-Risk Everyday*, *Medium-Risk New Region*, *High-Risk Anomaly*).
  - Risk gauge visualizer with real-time confidence scores and decision recommendations.
- **📁 Batch CSV Prediction Engine**:
  - Drag-and-drop CSV batch upload.
  - Built-in loader to evaluate sample records directly from `data/test_transaction.csv`.
  - Configurable decision threshold slider to tune precision/recall trade-offs.
  - Risk distribution analytics and one-click export of scored predictions (`fraud_predictions.csv`).
---
- **Domain**: Machine Learning / Financial Risk Intelligence / Fraud Analytics
- **Technologies**: Python, XGBoost, LightGBM, Scikit-Learn, Pandas, NumPy, Streamlit
