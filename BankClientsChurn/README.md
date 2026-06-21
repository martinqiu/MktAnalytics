# 🏦 Bank Client Churn Detection Dashboard

> Interactive machine learning dashboard for predicting customer churn —
> built with **Streamlit**, **scikit-learn**, **XGBoost**, and **LightGBM**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-≥1.40-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.4-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-≥2.0-green)
![LightGBM](https://img.shields.io/badge/LightGBM-≥4.0-yellow)

---

[View Analysis Summary (PDF)](bank_churn_summary.pdf)

---

## 📥 Step 1: Download or Clone

Before you can run the dashboard, you must have the files on your computer.

### **Option A: Download ZIP (Easiest)**
1.  Click the green **Code** button at the top of this GitHub page.
2.  Select **Download ZIP**.
3.  **IMPORTANT:** Once downloaded, right-click the ZIP file and select **"Extract All"**. Do not try to run the scripts from inside the zipped folder.

### **Option B: Clone with Git (Professional)**
1.  Open your Terminal or Command Prompt.
2.  Run the following command:
    ```bash
    git clone https://github.com/your-username/BankClientsChurn.git
    ```

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Total Clients | 10,127 |
| Features (IVs) | 19 |
| Attrited Clients | 1,627 |
| Churn Rate | 16.1% |
| Existing Clients | 8,500 |

**Target variable (DV):** `Status` — binary classification:
- 🔴 `Attrited Customer` — churned
- 🟢 `Existing Customer` — retained

---

## 🔢 Independent Variables (IVs) — per Data Dictionary

| # | Variable | Description |
|---|----------|-------------|
| 1 | `Age` | Client's age in years |
| 2 | `Gender` | Client's gender (M/F) |
| 3 | `Dependent_count` | Number of dependents |
| 4 | `Education_Level` | Highest education attained |
| 5 | `Marital_Status` | Single / Married / Divorced |
| 6 | `Income_Category` | Income bracket (e.g. <$40K) |
| 7 | `Card_Category` | Card type (Blue/Silver/Gold/Platinum) |
| 8 | `Tenure_in_Months` | Months as a customer |
| 9 | `Total_Products_Count` | Number of products held |
| 10 | `Recent Active_Mon` | Months since last activity |
| 11 | `Engagement_Count_12_mon` | Contacts in last 12 months |
| 12 | `Credit_Limit` | Credit card limit ($) |
| 13 | `Total_Revolving_Bal` | Unpaid balance carried forward |
| 14 | `Avg_Open_To_Buy` | Avg available credit (12 months) |
| 15 | `Total_Amt_Chng_Q4_Q1` | Change in transaction amount Q4→Q1 |
| 16 | `Total_Trans_Amt` | Total transaction amount (12 months) |
| 17 | `Total_Trans_Count` | Total transaction count (12 months) |
| 18 | `Total_Ct_Chng_Q4_Q1` | Change in transaction count Q4→Q1 |
| 19 | `Avg_Utilization_Ratio` | Avg card utilization ratio |

> 📌 `ID` is excluded (identifier, not a predictor).
> Categorical variables are one-hot encoded before model training.

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `bank_dashboard.py` | Main Streamlit dashboard (3 tabs) |
| `bank_competition.py` | ML competition script + PowerPoint generator |
| `Bank.clients.xlsx` | Dataset — 10,127 clients, 19 features + target |
| `bank_results.csv` | Pre-computed competition leaderboard *(auto-generated)* |
| `bank_tuning_results.csv` | Hyperparameter tuning results *(auto-generated)* |
| `Hyperprameter.md` | Hyperparameter reference notes (XGBoost, RF, LightGBM, SVM, KNN) |
| `requirements.txt` | Required Python libraries with versions |
| `install_and_run.bat` | First-time setup + launch *(Windows)* |
| `run_dashboard.bat` | Quick launch after initial setup |

---

## ⚙️ System Requirements

**Python 3.10 or higher** must be installed.
Download from [python.org/downloads](https://www.python.org/downloads/).
During installation, check **"Add Python to PATH"**.

### 📦 Required Libraries (from requirements.txt)

```
streamlit       >= 1.40.0
pandas          >= 2.0.0
numpy           >= 1.26.0
scikit-learn    >= 1.4.0
xgboost         >= 2.0.0
lightgbm        >= 4.0.0
plotly          >= 5.18.0
matplotlib      >= 3.8.0
seaborn         >= 0.13.0
python-pptx     >= 0.6.21
scipy           >= 1.11.0
openpyxl        >= 3.1.0
```

---

## 🚀 Getting Started (Windows)

**Step 1 — Install Python 3.10+** (check **"Add Python to PATH"**).

**Step 2 — First time**: double-click `install_and_run.bat`.

**Step 3 — Subsequent times**: double-click `run_dashboard.bat`.

---

## 🍎 Getting Started (Mac / Linux)

**Step 1 — Install Python 3.10+**.
**Step 2 — Open Terminal** and navigate to the project folder.
**Step 3 — Run the following command**:
```bash
sh setup_mac.sh
```

---

### 💻 Manual Launch (Any OS)

```bash
pip install -r requirements.txt
streamlit run bank_dashboard.py
```

---

## 📊 Dashboard Features

### Tab 1 — Summary Statistics
Dataset overview, churn distribution pie chart, categorical breakdowns
(by Gender, Education, Income, Card type, etc.), numeric feature distributions,
correlation heatmap, and data quality checks.

### Tab 2 — Model Competition 🏆
Leaderboard of 8 algorithms ranked by F1-score, interactive bar chart,
ROC curves for top 3 models, and confusion matrices.

### Tab 3 — Hyperparameter Tuning ⚙️
Select any of the 8 models, adjust hyperparameters with sliders,
click **🚀 Train Model** to see CV results, fold-by-fold performance,
confusion matrix, ROC curve, and feature importance — all updated live.
Each slider shows a contextual tip explaining what the parameter does
and its effect on overfitting/underfitting.

---

## 🏆 Model Competition — Top Results

> 8 classifiers evaluated via 5-fold stratified cross-validation.
> Class imbalance handled with `class_weight='balanced'` and XGBoost `scale_pos_weight`.

| Rank | Model | F1 Score | AUC-ROC |
|------|-------|----------|---------|
| 🥇 1 | Random Forest | ~0.93+ | ~0.97+ |
| 🥈 2 | LightGBM | ~0.92+ | ~0.97+ |
| 🥉 3 | XGBoost | ~0.92+ | ~0.97+ |
| 4 | Gradient Boosting | — | — |
| 5 | Logistic Regression | — | — |
| 6 | SVM (RBF) | — | — |
| 7 | K-Nearest Neighbors | — | — |
| 8 | Decision Tree | — | — |

> ℹ️ Exact scores are computed when you run `bank_competition.py`
> or open Tab 2 of the dashboard. Results are saved to `bank_results.csv`.

---

## ❓ Troubleshooting

**`streamlit: command not found`**
```bash
pip install streamlit
```

**`ModuleNotFoundError`**
```bash
pip install -r requirements.txt
```

**`No module named 'openpyxl'`**
```bash
pip install openpyxl
```
*(Required to read `.xlsx` files)*

**Port already in use**
```bash
streamlit run bank_dashboard.py --server.port 8502
```
Then open [http://localhost:8502](http://localhost:8502)

**Python not recognized**
Reinstall Python from [python.org/downloads](https://www.python.org/downloads/)
and check **"Add Python to PATH"** during setup.

---

## 🔁 How Cross-Validation Works

This project uses **5-fold stratified cross-validation** — not a single train/test split.

```
Full dataset (10,127 clients) split into 5 equal chunks:

Fold 1 → Test on Chunk 1,  Train on Chunks 2+3+4+5  (8,102 train / 2,025 test)
Fold 2 → Test on Chunk 2,  Train on Chunks 1+3+4+5
Fold 3 → Test on Chunk 3,  Train on Chunks 1+2+4+5
Fold 4 → Test on Chunk 4,  Train on Chunks 1+2+3+5
Fold 5 → Test on Chunk 5,  Train on Chunks 1+2+3+4

Final Score = average of 5 independent test scores
```

After CV identifies the best algorithm and hyperparameters,
the **final model is trained once on all 10,127 rows** — no holdout.
That single training produces the model estimates used for real predictions.

---

*Built with [Streamlit](https://streamlit.io) · [scikit-learn](https://scikit-learn.org) · [XGBoost](https://xgboost.readthedocs.io) · [LightGBM](https://lightgbm.readthedocs.io) · Python 3.10+*
