# Credit Card Fraud Detection — Walkthrough

## What Was Built

A complete end-to-end ML pipeline and interactive dashboard for credit card fraud detection.

---

## Dataset Summary

| Property | Value |
|----------|-------|
| File | `credit.Card.Fraud.csv` |
| Rows | 3,492 |
| Features | 29 (V1–V28 + Amount) |
| Target | `Class` → Fraud (496, 14.2%) / NonFraud (2,996, 85.8%) |
| Type | Binary classification |

---

## 🏆 Competition Leaderboard (5-Fold Stratified CV)

| Rank | Model | F1 Score | AUC-ROC | Accuracy | Precision | Recall |
|------|-------|----------|---------|----------|-----------|--------|
| 🥇 1 | **Random Forest** | **0.9886** | 0.9800 | 0.9802 | 0.9781 | **0.9993** |
| 🥈 2 | LightGBM | 0.9876 | 0.9809 | 0.9785 | 0.9806 | 0.9947 |
| 🥉 3 | XGBoost | 0.9874 | 0.9810 | 0.9782 | 0.9784 | 0.9967 |
| 4 | Gradient Boosting | 0.9863 | 0.9815 | 0.9762 | 0.9783 | 0.9943 |
| 5 | K-Nearest Neighbors | 0.9844 | 0.9532 | 0.9728 | 0.9711 | 0.9980 |
| 6 | SVM (RBF) | 0.9839 | 0.9822 | 0.9722 | 0.9811 | 0.9866 |
| 7 | Logistic Regression | 0.9811 | 0.9765 | 0.9676 | 0.9845 | 0.9776 |
| 8 | Decision Tree | 0.9749 | 0.9348 | 0.9573 | 0.9808 | 0.9693 |

> **Winner: Random Forest** with F1=0.9886, highest Recall (0.9993 — near-perfect fraud capture)

---

## Top 3 Models — Hyperparameter Tuning Results

| Model | Best CV F1 | Best Params |
|-------|-----------|-------------|
| Random Forest | 0.9886 | n_estimators=100, max_depth=None, min_samples_split=5 |
| LightGBM | 0.9881 | n_estimators=200, learning_rate=0.1, max_depth=6 |
| XGBoost | **0.9889** | n_estimators=200, learning_rate=0.2, max_depth=6 |

---

## Files Generated

| File | Description |
|------|-------------|
| `fraud_competition.py` | Main competition + PPTX generator |
| `fraud_dashboard.py` | Streamlit dashboard |
| `fraud_results.csv` | Full leaderboard |
| `fraud_tuning_results.csv` | Tuning results for top 3 |
| `fraud_summary.pptx` | 8-slide PowerPoint |
| `fraud_plots/` | ROC curves, confusion matrices, feature importances |

---

## Dashboard

Run with:
```bash
streamlit run fraud_dashboard.py
```
URL: **http://localhost:8501**
