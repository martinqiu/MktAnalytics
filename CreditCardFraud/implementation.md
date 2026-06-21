# Credit Card Fraud Classification — Implementation Summary

This project implements a full machine learning competition and interactive dashboard for detecting fraudulent credit card transactions.

## Dataset Overview

- **Dataset**: `credit.Card.Fraud.csv`
- **Size**: 3,492 rows
- **Features**: 29 independent variables (V1–V28 PCA features + Amount).
- **Target**: `Class` (Fraud / Non-Fraud).
- **Class Imbalance**: Approximately 14.2% Fraud.

## Technical Implementation

### 1. Algorithms Evaluated
Eight classifiers were tested using **5-fold stratified cross-validation** to handle class imbalance:
- **Random Forest** (Winner)
- **LightGBM**
- **XGBoost**
- **Gradient Boosting**
- **K-Nearest Neighbors**
- **SVM (RBF)**
- **Logistic Regression**
- **Decision Tree**

### 2. Implementation details
- **Evaluation Metric**: Primary focus on **F1-Score** and **Recall** to minimize missed fraud cases.
- **Handling Imbalance**: Implemented `class_weight='balanced'` for scikit-learn models and `scale_pos_weight` for XGBoost.
- **Reporting**: Automated generation of an 8-slide PowerPoint summary (`fraud_summary.pptx`) and a leaderboard CSV (`fraud_results.csv`).

### 3. Streamlit Dashboard
The dashboard (`fraud_dashboard.py`) includes:
- **Tab 1 — Summary Stats**: Dataset overview, class distribution pie charts, interactive feature histograms, and correlation heatmaps.
- **Tab 2 — Model Competition**: Leaderboard rankings, metrics comparison, ROC curves, and confusion matrices for top-performing models.
- **Tab 3 — Hyperparameter Tuning**: Manual control over hyperparameters with real-time CV performance tracking.

## Deliverables
- `fraud_competition.py`: Pre-configured competition script.
- `fraud_dashboard.py`: Interactive user interface.
- `fraud_summary.pptx`: Professional report.
- `fraud_results.csv`: Evaluation metrics for all models.
