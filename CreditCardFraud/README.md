# Credit Card Fraud Detection Dashboard

An interactive machine learning dashboard for credit card fraud detection —
built with Streamlit, scikit-learn, XGBoost, and LightGBM.

[View Analysis Summary (PDF)](fraud_analysis_summary.pdf)

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
    git clone https://github.com/your-username/CreditCardFraud.git
    ```

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `fraud_dashboard.py` | Main Streamlit dashboard |
| `fraud_competition.py` | ML competition + PowerPoint generator |
| `credit.Card.Fraud.csv` | Dataset (3,492 transactions) |
| `fraud_results.csv` | Pre-computed competition leaderboard |
| `fraud_tuning_results.csv` | Hyperparameter tuning results |
| `requirements.txt` | Required Python libraries |
| `install_and_run.bat` | First-time setup + launch (Windows) |
| `run_dashboard.bat` | Launch only (after setup) |

---

## ⚙️ System Requirements

**Python 3.10 or higher** must be installed.
Download from [python.org/downloads](https://www.python.org/downloads/).
During installation, check **"Add Python to PATH"**.


### 📦 Required Libraries (from requirements.txt)

```
streamlit>=1.40.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.0.0
plotly>=5.18.0
matplotlib>=3.8.0
seaborn>=0.13.0
python-pptx>=0.6.21
scipy>=1.11.0
openpyxl>=3.1.0
```

---

## 🚀 Getting Started (Windows)

**Step 1 — Install Python 3.10+** (check **"Add Python to PATH"**).

**Step 2 — First time**: double-click `install_and_run.bat`.

**Step 3 — Subsequent times**: double-click `run_dashboard.bat`.

---

### 🍎 Getting Started (Mac / Linux)

#### First Time & Launch
1. Open your **Terminal** app.
2. Drag the project folder into the terminal or use `cd` to move into it.
3. Type the following command and press Enter:
```bash
sh setup_mac.sh
```

---

### 💻 Manual Launch (Any OS)
```bash
pip install -r requirements.txt
streamlit run fraud_dashboard.py
```

---

## 📊 Dashboard Features

| Tab | Content |
|-----|---------|
| 📊 Summary Statistics | Dataset overview, feature distributions, correlation heatmap |
| 🏆 Model Competition | Leaderboard of 8 algorithms ranked by F1-score |
| ⚙️ Hyperparameter Tuning | Interactive model training with adjustable parameters |

---

## 🏆 Competition Results (pre-computed)

| Rank | Model | F1 Score | AUC-ROC |
|------|-------|----------|---------|
| 🥇 1 | Random Forest | 0.9886 | 0.9800 |
| 🥈 2 | LightGBM | 0.9876 | 0.9809 |
| 🥉 3 | XGBoost | 0.9874 | 0.9810 |

---



## ❓ Troubleshooting

**`streamlit: command not found`**
→ Run: `pip install streamlit`

**`ModuleNotFoundError`**
→ Run: `pip install -r requirements.txt`

**Port already in use**
→ Run: `streamlit run fraud_dashboard.py --server.port 8502`
   Then open: http://localhost:8502

**Python not recognized**
→ Reinstall Python and check "Add Python to PATH" during setup
