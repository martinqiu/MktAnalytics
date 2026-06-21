# Bank Client Churn Prediction — Implementation Summary

This project implements a complete machine learning pipeline and interactive dashboard to identify clients at risk of leaving the bank (Churn).

## Dataset Overview

- **Source**: `Bank.clients.xlsx`
- **Size**: 10,127 rows, 21 columns
- **Target (DV)**: `Status` (Attrited Customer / Existing Customer)
- **Features (IVs)**: 19 predictors including Age, Gender, Education, Income, Credit Limit, and Transaction patterns.

## Technical Implementation

### 1. Preprocessing & Encoding
- **Categorical Handling**: `Gender`, `Education_Level`, `Marital_Status`, `Income_Category`, and `Card_Category` were processed using one-hot encoding (`pd.get_dummies`).
- **Class Imbalance**: The dataset is imbalanced (16% churn). This was addressed using `class_weight='balanced'` for standard algorithms and `scale_pos_weight` for XGBoost.
- **Scaling**: Numeric features were standardized within model pipelines where appropriate (e.g., for SVM, KNN, and Logistic Regression).

### 2. Model Competition
Eight classification algorithms were evaluated using **5-fold stratified cross-validation**:
- **Gradient Boosting** (Top Performer)
- **LightGBM**
- **XGBoost**
- **Random Forest**
- **Decision Tree**
- **SVM (RBF)**
- **K-Nearest Neighbors**
- **Logistic Regression**

### 3. Streamlit Dashboard
The dashboard (`bank_dashboard.py`) provides an interactive interface for model exploration:

- **Tab 1: Summary Statistics**: Visualizes churn distribution, categorical breakdowns, and numeric feature correlations.
- **Tab 2: Model Competition**: Presents the leaderboard, interactive metric comparisons, and ROC curves for the top 3 models.
- **Tab 3: Hyperparameter Tuning**: Allows live tuning of all 8 models with a per-parameter "Tip Card" system based on documented best practices.

## Execution
- **Full Competition**: Run `python bank_competition.py` to generate the leaderboard and a professional 8-slide PowerPoint summary (`bank_summary.pptx`).
- **Dashboard**: Run `streamlit run bank_dashboard.py` (or use the provided `.bat` files for one-click launch).
