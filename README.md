# 👑 Haveli — Customer Purchase Prediction

### Where ancient wisdom meets modern intelligence

A luxury-themed Machine Learning web app that predicts whether a customer will purchase a product based on their **Age** and **Estimated Salary**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📖 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [Features](#4-features)
5. [Technologies Used](#5-technologies-used)
6. [Project Workflow](#6-project-workflow)
7. [Data Preprocessing](#7-data-preprocessing)
8. [Models Used](#8-models-used)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Model Comparison](#10-model-comparison)
11. [Overfitting Analysis](#11-overfitting-analysis)
12. [Cross-Validation](#12-cross-validation)
13. [Hyperparameter Tuning](#13-hyperparameter-tuning)
14. [Prediction Example](#14-prediction-example)
15. [Streamlit App — Haveli Premium](#15-streamlit-app--haveli-premium)
16. [Project Structure](#16-project-structure)
17. [How to Run](#17-how-to-run)
18. [Results](#18-results)
19. [Screenshots](#19-screenshots)
20. [Limitations](#20-limitations)
21. [Future Improvements](#21-future-improvements)
22. [Author](#22-author)

---

## 1. Project Overview

**Haveli** is a complete end-to-end Machine Learning project that:

- Trains and compares **four classification models** on customer data
- Predicts whether a customer will purchase a product (**Buy = 1 / No Buy = 0**)
- Ships with a **luxury-themed Streamlit web app** for interactive predictions
- Includes **batch prediction**, **feature insights**, and **session history**

The theme — a **Rajasthani palace aesthetic** — brings a premium, memorable touch to an otherwise standard ML dashboard: midnight-blue surfaces, royal gold accents, and glowing diya animations.

---

## 2. Problem Statement

> **Given a customer's `Age` and `EstimatedSalary`, predict whether they will purchase the product.**

This is a **binary classification** problem:

- **`0`** → Customer will **not** buy
- **`1`** → Customer **will** buy

**Why it matters:** Businesses waste money targeting non-buyers. A reliable prediction helps prioritize marketing spend and personalize outreach.

---

## 3. Dataset

**File:** `data/Social_Network_Ads.csv`
**Source:** Classic UCI-derived social network ads dataset
**Size:** ~400 customer records
**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `User ID` | int | Unique identifier (not used) |
| `Gender` | str | Male / Female (not used in v1) |
| `Age` | int | Customer age |
| `EstimatedSalary` | int | Customer's estimated annual salary |
| `Purchased` | int | **Target** — 0 or 1 |

> ⚠️ `User ID` and `Gender` are intentionally excluded in this version. Feature engineering on those is a future improvement.

---

## 4. Features

### Input Features (`X`)
- **`Age`** — numeric
- **`EstimatedSalary`** — numeric

### Target (`y`)
- **`Purchased`** — binary (0 / 1)

---

## 5. Technologies Used

| Category | Libraries |
|----------|-----------|
| Data | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| ML | `scikit-learn` |
| Model Persistence | `joblib` |
| Web App | `streamlit` |
| Notebook | `jupyter` |

---

## 6. Project Workflow

```
Raw Data
   ↓
EDA  (explore, visualize)
   ↓
Data Cleaning  (nulls, dupes, dtypes)
   ↓
Feature / Target split
   ↓
Stratified Train/Test Split (80 / 20)
   ↓
Feature Scaling  (StandardScaler — for LR & KNN only)
   ↓
Train 4 Models  (LR, DT, RF, KNN)
   ↓
Evaluate  (Accuracy, Precision, Recall, F1, ROC-AUC)
   ↓
Confusion Matrix + ROC Curves
   ↓
Cross-Validation (5-fold StratifiedKFold)
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Final Evaluation on untouched test set
   ↓
Feature Importance / Interpretation
   ↓
Save Models & Scaler  (.pkl)
   ↓
Streamlit App  (Haveli Premium)
```

---

## 7. Data Preprocessing

- ✅ No missing values
- ✅ No duplicate rows
- ✅ `Age` and `EstimatedSalary` used as-is (already numeric)
- ✅ **Stratified** 80/20 train/test split (`stratify=y`) — preserves class ratio in both sets
- ✅ **StandardScaler** applied only for **Logistic Regression** and **KNN**
  - Scaler is **fit on train only**, then applied to test → prevents **data leakage**
- ✅ Tree-based models (Decision Tree, Random Forest) use **unscaled** features — they split on thresholds, so scaling doesn't change their decisions

---

## 8. Models Used

| # | Model | Why It's Included |
|---|-------|-------------------|
| 1 | **Logistic Regression** | Baseline linear classifier; interpretable coefficients |
| 2 | **Decision Tree** | Non-linear, easy to visualize; prone to overfitting |
| 3 | **Random Forest** | Ensemble of trees; usually robust and accurate |
| 4 | **K-Nearest Neighbors** | Instance-based; sensitive to scaling — good contrast |

No single model is assumed "best" — the project compares all four empirically.

---

## 9. Evaluation Metrics

| Metric | What It Tells Us |
|--------|------------------|
| **Accuracy** | Overall % correct (misleading when classes are imbalanced) |
| **Precision** | Of predicted buyers, how many actually bought |
| **Recall** | Of actual buyers, how many we caught |
| **F1-Score** | Harmonic mean of precision & recall |
| **ROC-AUC** | Ranking quality — probability model ranks a random buyer above a random non-buyer |

**Confusion Matrix terms:**
- **TP** — predicted Buy, actually bought ✅
- **TN** — predicted No Buy, actually didn't buy ✅
- **FP** — predicted Buy, actually didn't buy (wasted budget) ❌
- **FN** — predicted No Buy, actually bought (missed revenue) ❌

---

## 10. Model Comparison

The notebook produces a comparison table with all metrics calculated programmatically — **no manually typed scores**.

| Model | Train Acc | Test Acc | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|----------|-----------|--------|----|---------|
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| KNN | — | — | — | — | — | — |

> 📌 Fill this table with **your own results** after running the notebook — this keeps the README honest and reproducible.

---

## 11. Overfitting Analysis

For each model we compute:

```
Gap = Train Accuracy − Test Accuracy
```

**Interpretation:**

| Situation | Diagnosis |
|-----------|-----------|
| Small gap (≤ 0.03) | Good generalization |
| Large gap (> 0.10) | Overfitting |
| Both scores low | Underfitting |

**Typical pattern on this dataset:**
- **Decision Tree** → largest train/test gap (overfits)
- **Logistic Regression & Random Forest** → tighter gap
- **KNN** → sensitive to `k` value

---

## 12. Cross-Validation

We use **5-fold Stratified K-Fold** on the training set:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
```

**Why CV matters:**
- One train/test split can be lucky or unlucky
- K-Fold gives **K scores** → mean + std = stable estimate
- `Stratified` keeps the class ratio in every fold

---

## 13. Hyperparameter Tuning

**GridSearchCV** exhaustively tries combinations of hyperparameters using **cross-validation on the training set only** (no leakage).

| Model | Hyperparameters tuned |
|-------|-----------------------|
| Logistic Regression | `C`, `solver` |
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| KNN | `n_neighbors`, `weights`, `metric` |

Best parameters and best cross-validated AUC are printed in the notebook.

---

## 14. Prediction Example

```python
from app import predict_customer   # or from your own helper

pred, proba = predict_customer(age=30, salary=60000)

print(pred)    # 1  → will buy
print(proba)   # 0.87 → 87% probability
```

**The app also supports batch prediction:**

Upload a CSV with `Age` and `EstimatedSalary` columns → download results with `Prediction`, `Probability`, and `Persona` appended.

---

## 15. Streamlit App — Haveli Premium

A luxury dark-themed dashboard with a **Rajasthani palace aesthetic**.

### ✨ Features

| Feature | Description |
|---------|-------------|
| 👑 **Premium dark theme** | Midnight backgrounds, royal gold accents, ivory text |
| 🕯️ **Animated diya** | Flickering glow in the header |
| 🔮 **Single prediction** | Age + Salary → prediction + probability gauge |
| 📂 **Batch prediction** | Upload CSV → download predictions |
| 📊 **Insights tab** | Feature importance, class distribution, scatter plot |
| 📜 **History tab** | Session log of predictions with CSV export |
| 🎭 **Buyer personas** | Auto-classified: Young Explorer, Rising Professional, etc. |
| ⚙️ **Model selector** | Switch between all 4 models in the sidebar |

### 🎨 Color Palette

| Role | Color |
|------|-------|
| Background | `#0F1419` Midnight |
| Cards | `#1A2035` Indigo → `#232B45` Velvet |
| Primary accent | `#E8B84B` Gold |
| Body text | `#F5EFE0` Ivory |
| Success | `#2E8B57` Emerald |
| Warning | `#8B1E1E` Deep Maroon |

Every text/background pair passes **WCAG AA or AAA** contrast.

### 🚀 Run the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

---

## 16. Project Structure

```
customer-purchase-prediction/
│
├── assets/
│   └── style.css                     # Haveli Premium theme (CSS)
│
├── data/
│   └── Social_Network_Ads.csv        # Raw dataset
│
├── models/
│   ├── logistic_model.pkl            # Tuned Logistic Regression
│   ├── decision_tree_model.pkl       # Tuned Decision Tree
│   ├── random_forest_model.pkl       # Tuned Random Forest
│   ├── knn_model.pkl                 # Tuned KNN
│   └── scaler.pkl                    # Fitted StandardScaler
│
├── notebooks/
│   └── customer_prediction.ipynb     # Full ML workflow
│
├── app.py                            # Streamlit app (Haveli Premium)
├── check_env.py                      # Environment & model sanity checker
├── README.md                         # You are here
├── requirements.txt                  # Python dependencies
└── .gitignore
```

---

## 17. How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/customer-purchase-prediction.git
cd customer-purchase-prediction
```

### 2️⃣ Create a fresh environment (recommended)

```bash
conda create -n haveli python=3.11 -y
conda activate haveli
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Verify environment & models

```bash
python check_env.py
```

You should see ✅ for every `.pkl` file. If anything is ❌ or ⚠️, retrain via the notebook first.

### 5️⃣ (Re)train the models

```bash
jupyter notebook notebooks/customer_prediction.ipynb
```

Run all cells → models are saved to `models/`.

### 6️⃣ Launch the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 18. Results

### 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| KNN | — | — | — | — | — |

> 📌 **Fill this in from your own notebook run.** Never fabricate scores — they depend on your split and tuning.

### 📊 Key Observations

- Tree-based models handle non-linear Age/Salary boundaries well
- Logistic Regression provides the most interpretable coefficients
- KNN's performance depends heavily on `k` and proper scaling
- No single model dominates every metric

---


## 19. Limitations

- 📉 **Small dataset** (~400 rows) — results may not generalize to real production data
- 🧩 **Only 2 features used** (`Age`, `EstimatedSalary`)
- 🚫 `Gender` and `User ID` excluded in v1
- 🎯 Predictions are probabilistic — not guarantees
- 🧪 Class imbalance is mild but not corrected (no SMOTE etc.)

---

## 20. Future Improvements

- 🚢 Port the pipeline to the **Titanic dataset** for richer messiness (missing values, categorical features)
- 🛠️ **Feature engineering** — age buckets, salary bands, interaction terms
- ⚖️ **Class imbalance handling** — SMOTE, class weights
- 🌲 **Gradient boosting** — XGBoost, LightGBM, CatBoost
- 🔍 **SHAP explanations** for model interpretability
- 🐳 **Dockerize** the app
- ☁️ **Deploy to Streamlit Cloud** with a public URL
- 🔐 Add **user authentication** for multi-tenant use
- 📈 Add **prediction confidence distribution** and drift detection

---

## 21. Author

**A.R. Khan**
Built with ❤️ using Streamlit, Scikit-learn & a touch of heritage.

> ⭐ If you found this project useful, consider giving it a star on GitHub!

---

<div align="center">

**👑 Haveli ML Premium 👑**

*Where ancient wisdom meets modern intelligence*

</div>
