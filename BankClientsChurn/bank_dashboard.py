"""
Bank Client Churn — Streamlit Dashboard
========================================
Dataset : bank.clients.xlsx (sheet: bank.churners)
DV      : Status (Existing Customer vs Attrited Customer)
Tab 1: Summary Statistics
Tab 2: Model Competition Results
Tab 3: Hyperparameter Tuning (interactive)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Client Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%); }
    .main .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #16213e, #0f3460);
        border: 1px solid #00b4d8; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        box-shadow: 0 4px 20px rgba(0,180,216,0.15); margin: 0.3rem 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #00b4d8; }
    .metric-label { font-size: 0.85rem; color: #adb5bd; text-transform: uppercase; letter-spacing: 1px; }
    .metric-delta { font-size: 0.9rem; color: #06d6a0; }
    .section-header {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 1.6rem; font-weight: 800; margin: 1rem 0 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #0f3460; border-radius: 8px 8px 0 0;
        padding: 10px 24px; color: #adb5bd; font-weight: 600;
        border: 1px solid #1a4a7a;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0077b6, #00b4d8) !important;
        color: white !important;
    }
    .info-box {
        background: rgba(0,180,216,0.1); border-left: 4px solid #00b4d8;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.5rem 0;
    }
    .stSlider label { color: #e0e0e0 !important; }
    .stSelectbox label { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
DATA_FILE    = BASE / "bank.clients.xlsx"
RESULTS_FILE = BASE / "bank_results.csv"
RANDOM_STATE = 42
N_SPLITS     = 5
TARGET       = "Status"
CAT_COLS     = ["Gender", "Education_Level", "Marital_Status",
                "Income_Category", "Card_Category"]

COLORS = {
    "primary": "#00b4d8", "danger": "#ff6b6b", "success": "#06d6a0",
    "warning": "#f4a261", "gold": "#ffd700", "silver": "#c0c0c0",
    "bronze": "#cd7f32",
}

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = pd.read_excel(DATA_FILE, sheet_name="bank.churners")
    df_raw.columns = df_raw.columns.str.strip()
    df = df_raw.drop(columns=["ID"]).copy()
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET])
    X_raw = df.drop(columns=[TARGET])
    X = pd.get_dummies(X_raw, columns=CAT_COLS, drop_first=False)
    return df_raw, df, X, y, le

@st.cache_data
def load_results():
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE, index_col=0)
    return None

# ── Classifier Factory ────────────────────────────────────────────────────────
def build_classifiers(scale_pos_weight=1.0):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                       class_weight="balanced"))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            class_weight="balanced", random_state=RANDOM_STATE, verbose=-1),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, n_jobs=-1))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE))
        ]),
    }

# ── Load Data ─────────────────────────────────────────────────────────────────
df_raw, df, X, y, le = load_data()

n_classes      = len(np.unique(y))
scoring_f1     = "f1"        if n_classes == 2 else "f1_macro"
scoring_prec   = "precision" if n_classes == 2 else "precision_macro"
scoring_rec    = "recall"    if n_classes == 2 else "recall_macro"

attrited_idx   = list(le.classes_).index("Attrited Customer")
existing_idx   = 1 - attrited_idx
churn_count    = int((y == attrited_idx).sum())
existing_count = int((y == existing_idx).sum())
scale_pos_wt   = existing_count / max(churn_count, 1)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <span style='font-size:2.5rem;'>🏦</span>
        <h3 style='color:#00b4d8; margin:0;'>Client Churn</h3>
        <p style='color:#adb5bd; font-size:0.85rem;'>ML Competition Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**📁 Dataset:** `{DATA_FILE.name}`")
    st.markdown(f"**📊 Clients:** `{df_raw.shape[0]:,}`")
    st.markdown(f"**📐 Features (encoded):** `{X.shape[1]}`")
    st.markdown(f"**🎯 Churn Rate:** `{churn_count/len(y):.2%}`")
    # (Navigation section removed to avoid redundancy with top tabs)
    st.markdown("---")
    st.markdown("<p style='color:#adb5bd;font-size:0.75rem;text-align:center;'>5-Fold Stratified Cross-Validation</p>",
                unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1rem 0 0.5rem 0;'>
    <span style='font-size:3rem;'>🏦</span>
    <h1 style='color:#00b4d8;font-size:2.5rem;font-weight:900;margin:0;letter-spacing:-1px;'>
        Bank Client Churn Detection
    </h1>
    <p style='color:#adb5bd;font-size:1.1rem;margin:0.3rem 0 1rem 0;'>
        Machine Learning Classification Competition Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, delta in [
    (c1, "Total Clients",    f"{df_raw.shape[0]:,}", ""),
    (c2, "Features (IVs)",   f"{X.shape[1]}",        "encoded"),
    (c3, "Attrited Clients", f"{churn_count:,}",     f"{churn_count/len(y):.2%} churn rate"),
    (c4, "Existing Clients", f"{existing_count:,}",  f"{existing_count/len(y):.2%} rate"),
    (c5, "Classes",          f"{n_classes}",          "Binary classification"),
]:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-delta'>{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Summary Statistics",
    "🏆  Model Competition",
    "⚙️  Hyperparameter Tuning",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_pie, col_bar = st.columns([1, 2])

    with col_pie:
        status_counts = df_raw[TARGET].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=status_counts.index.tolist(),
            values=status_counts.values.tolist(),
            hole=0.55,
            marker=dict(colors=[COLORS["danger"], COLORS["success"]],
                        line=dict(color="#0d1b2a", width=3)),
            textinfo="percent+label",
            textfont=dict(size=13, color="white"),
        ))
        fig_pie.update_layout(
            title=dict(text="Churn Distribution", font=dict(size=16, color="#00b4d8")),
            paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            showlegend=True, height=320, margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        # Categorical breakdown
        cat_choice = st.selectbox("Explore by category:", CAT_COLS)
        cat_counts = df_raw.groupby([cat_choice, TARGET]).size().reset_index(name="Count")
        fig_cat = px.bar(
            cat_counts, x=cat_choice, y="Count", color=TARGET,
            barmode="group",
            color_discrete_map={"Attrited Customer": COLORS["danger"],
                                 "Existing Customer": COLORS["success"]},
            template="plotly_dark", height=320,
        )
        fig_cat.update_layout(
            title=dict(text=f"Churn by {cat_choice}", font=dict(size=15, color="#00b4d8")),
            paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            margin=dict(t=50, b=10), legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Numeric feature distribution
    st.markdown("<div class='section-header'>Numeric Feature Distributions</div>",
                unsafe_allow_html=True)
    num_cols_raw = [c for c in df.drop(columns=[TARGET]).columns if c not in CAT_COLS]
    feat_sel = st.selectbox("Select numeric feature:", num_cols_raw, key="feat_dist")
    fig_hist = go.Figure()
    for status in df_raw[TARGET].unique():
        vals = df_raw[df_raw[TARGET] == status][feat_sel]
        fig_hist.add_trace(go.Histogram(
            x=vals, name=status, opacity=0.75, nbinsx=50,
            marker_color=COLORS["danger"] if "Att" in status else COLORS["success"],
        ))
    fig_hist.update_layout(
        barmode="overlay",
        title=dict(text=f"Distribution of {feat_sel} by Status",
                   font=dict(size=15, color="#00b4d8")),
        xaxis=dict(title=feat_sel, gridcolor="#1a2a3a"),
        yaxis=dict(title="Count", gridcolor="#1a2a3a"),
        paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        height=320, margin=dict(t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Descriptive stats
    st.markdown("<div class='section-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
    num_df = df_raw[[c for c in num_cols_raw]].describe().T.round(4)
    st.dataframe(
        num_df.style.background_gradient(subset=["mean", "std"], cmap="Blues")
                    .format("{:.4f}"),
        height=320, use_container_width=True,
    )

    # Correlation heatmap (numeric only)
    st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
    corr_sel = st.multiselect("Select features:", num_cols_raw,
                               default=num_cols_raw[:10], key="corr_feat")
    if corr_sel:
        corr_m = df_raw[corr_sel].corr()
        fig_hm = go.Figure(go.Heatmap(
            z=corr_m.values, x=corr_m.columns, y=corr_m.index,
            colorscale="RdBu_r", zmid=0,
            text=corr_m.values.round(2), texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="r", tickfont=dict(color="white")),
        ))
        fig_hm.update_layout(
            title=dict(text="Pearson Correlation Matrix", font=dict(size=15, color="#00b4d8")),
            paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            height=480, margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # Data quality
    st.markdown("<div class='section-header'>Data Quality</div>", unsafe_allow_html=True)
    mv = df_raw.isnull().sum()
    c_mv1, c_mv2 = st.columns(2)
    with c_mv1:
        st.markdown(f"""
        <div class='info-box'>
            <strong style='color:#00b4d8;'>Missing Values:</strong>
            <span style='color:#e0e0e0;'> {mv.sum()} total
            ({mv[mv > 0].shape[0]} columns affected)</span>
        </div>""", unsafe_allow_html=True)
    with c_mv2:
        st.markdown(f"""
        <div class='info-box'>
            <strong style='color:#00b4d8;'>Duplicates:</strong>
            <span style='color:#e0e0e0;'> {df_raw.duplicated().sum()} duplicate rows</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPETITION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Classification Competition Leaderboard</div>",
                unsafe_allow_html=True)

    results_df = load_results()

    if results_df is None:
        st.info("No pre-computed results found. Running quick competition (may take a few minutes)...")
        with st.spinner("🔄 Running 8-model competition..."):
            clf_dict = build_classifiers(scale_pos_wt)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            comp_results = []
            prog = st.progress(0)
            for i, (name, clf) in enumerate(clf_dict.items()):
                f1_s    = cross_val_score(clf, X, y, cv=skf, scoring=scoring_f1,   n_jobs=-1)
                auc_s   = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc",    n_jobs=-1)
                acc_s   = cross_val_score(clf, X, y, cv=skf, scoring="accuracy",   n_jobs=-1)
                prec_s  = cross_val_score(clf, X, y, cv=skf, scoring=scoring_prec, n_jobs=-1)
                rec_s   = cross_val_score(clf, X, y, cv=skf, scoring=scoring_rec,  n_jobs=-1)
                comp_results.append({
                    "Model": name,
                    "F1 (mean)": f1_s.mean(),   "F1 (std)": f1_s.std(),
                    "AUC (mean)": auc_s.mean(), "AUC (std)": auc_s.std(),
                    "Accuracy (mean)": acc_s.mean(),
                    "Precision (mean)": prec_s.mean(),
                    "Recall (mean)": rec_s.mean(),
                })
                prog.progress((i + 1) / len(clf_dict))
            results_df = (pd.DataFrame(comp_results)
                          .sort_values("F1 (mean)", ascending=False)
                          .reset_index(drop=True))
            results_df.index += 1

    top3_names  = results_df["Model"].head(3).tolist()
    medals      = {"1": "🥇", "2": "🥈", "3": "🥉"}

    col_lb, col_metrics = st.columns([1.5, 1])

    with col_lb:
        lb_display = results_df[["Model","F1 (mean)","F1 (std)","AUC (mean)","AUC (std)",
                                  "Accuracy (mean)","Precision (mean)","Recall (mean)"]].copy()
        lb_display.insert(0, "Rank", [medals.get(str(i), f"#{i}") for i in results_df.index])
        st.dataframe(
            lb_display.style
                .format({"F1 (mean)": "{:.4f}", "F1 (std)": "{:.4f}",
                         "AUC (mean)": "{:.4f}", "AUC (std)": "{:.4f}",
                         "Accuracy (mean)": "{:.4f}",
                         "Precision (mean)": "{:.4f}", "Recall (mean)": "{:.4f}"})
                .background_gradient(subset=["F1 (mean)"], cmap="YlOrRd"),
            height=330, use_container_width=True,
        )

    with col_metrics:
        best = results_df.iloc[0]
        st.markdown(f"""
        <div style='text-align:center;margin-bottom:1rem;'>
            <span style='color:{COLORS["gold"]};font-size:2.5rem;'>🥇</span>
            <h3 style='color:{COLORS["gold"]};margin:0;'>{best["Model"]}</h3>
            <p style='color:#adb5bd;margin:0;'>Best Model</p>
        </div>""", unsafe_allow_html=True)
        for lbl, key in [("F1 Score", "F1 (mean)"), ("AUC-ROC", "AUC (mean)"),
                          ("Accuracy", "Accuracy (mean)"), ("Precision", "Precision (mean)"),
                          ("Recall", "Recall (mean)")]:
            st.markdown(f"""
            <div class='metric-card' style='margin:0.2rem 0;padding:0.6rem;'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='font-size:1.5rem;'>{best[key]:.4f}</div>
            </div>""", unsafe_allow_html=True)

    # Bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    metric_choice = st.selectbox("Compare models by:",
        ["F1 (mean)", "AUC (mean)", "Accuracy (mean)", "Precision (mean)", "Recall (mean)"])
    color_vals = [COLORS["gold"] if i == 0 else COLORS["silver"] if i == 1
                  else COLORS["bronze"] if i == 2 else COLORS["primary"]
                  for i in range(len(results_df))]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=results_df["Model"], y=results_df[metric_choice],
        marker_color=color_vals,
        text=results_df[metric_choice].round(4),
        textposition="outside", textfont=dict(color="white", size=11),
    ))
    fig_bar.update_layout(
        title=dict(text=f"Model Comparison — {metric_choice}",
                   font=dict(size=16, color="#00b4d8")),
        xaxis=dict(title="Model", gridcolor="#1a2a3a", linecolor="#2a3a4a"),
        yaxis=dict(title=metric_choice,
                   range=[0, min(1.15, results_df[metric_choice].max() + 0.08)],
                   gridcolor="#1a2a3a", linecolor="#2a3a4a"),
        paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        height=420, margin=dict(t=60, b=10), showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ROC + Confusion Matrix
    st.markdown("<div class='section-header'>Top 3 Models — Detailed Analysis</div>",
                unsafe_allow_html=True)
    with st.spinner("Fitting top 3 models..."):
        clf_dict = build_classifiers(scale_pos_wt)
        top3_details = {}
        for name in top3_names:
            clf = clf_dict[name]
            clf.fit(X, y)
            y_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None
            cm_val = confusion_matrix(y, clf.predict(X))
            auc_v  = roc_auc_score(y, y_prob) if y_prob is not None else None
            top3_details[name] = {"clf": clf, "cm": cm_val, "y_prob": y_prob, "auc": auc_v}

    col_roc, col_cm = st.columns([1, 1])
    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#adb5bd", dash="dash", width=1.5), name="Random",
        ))
        for (name, det), color in zip(top3_details.items(),
                                       [COLORS["gold"], COLORS["silver"], COLORS["bronze"]]):
            if det["y_prob"] is not None:
                fpr, tpr, _ = roc_curve(y, det["y_prob"])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={det['auc']:.4f})",
                    line=dict(color=color, width=2.5),
                ))
        fig_roc.update_layout(
            title=dict(text="ROC Curves — Top 3 Models", font=dict(size=15, color="#00b4d8")),
            xaxis=dict(title="False Positive Rate", gridcolor="#1a2a3a"),
            yaxis=dict(title="True Positive Rate", gridcolor="#1a2a3a"),
            paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            height=400, margin=dict(t=60, b=10),
            legend=dict(x=0.55, y=0.15, bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        sel_cm = st.selectbox("Select model for confusion matrix:", top3_names, key="cm_sel")
        det = top3_details[sel_cm]
        cm_v   = det["cm"]
        pct_v  = cm_v.astype(float) / cm_v.sum(axis=1, keepdims=True) * 100
        ann    = [[f"{v}<br>({p:.1f}%)" for v, p in zip(rv, rp)]
                  for rv, rp in zip(cm_v, pct_v)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_v, x=[f"Pred: {c}" for c in le.classes_],
            y=[f"Actual: {c}" for c in le.classes_],
            text=ann, texttemplate="%{text}", textfont=dict(size=13),
            colorscale="YlOrRd", showscale=True,
        ))
        fig_cm.update_layout(
            title=dict(text=f"Confusion Matrix — {sel_cm}",
                       font=dict(size=14, color="#00b4d8")),
            paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            xaxis=dict(gridcolor="#1a2a3a"), yaxis=dict(gridcolor="#1a2a3a"),
            height=380, margin=dict(t=60, b=30),
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Interactive Hyperparameter Tuning</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Adjust hyperparameters and click <strong>🚀 Train Model</strong> to evaluate
        performance via 5-fold stratified cross-validation.
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        model_choice = st.selectbox(
            "🤖 Select Model",
            ["Logistic Regression", "Decision Tree", "Random Forest",
             "Gradient Boosting", "XGBoost", "LightGBM",
             "K-Nearest Neighbors", "SVM (RBF)"],
            key="ht_model",
        )
        st.markdown(f"**Configure: {model_choice}**")
        st.markdown("---")

        # ── Param tips ────────────────────────────────────────────────────────
        PARAM_TIPS = {
            "n_estimators":      ("🌲", "Number of trees built by the ensemble.",
                                  "↑ More trees → higher accuracy, longer training.",
                                  "↓ Fewer trees → faster but may underfit.", None),
            "learning_rate":     ("🐢", "Shrinks each tree's contribution before adding it.",
                                  "↑ Higher rate → fast convergence but overfitting risk.",
                                  "↓ Lower rate → more robust; needs more trees.",
                                  "📌 Lower rate + more trees = almost always better accuracy."),
            "max_depth":         ("📐", "Maximum levels each tree can grow.",
                                  "↑ Deeper → captures complex patterns, ↑ overfit risk.",
                                  "↓ Shallower → simpler, more generalizable.", None),
            "subsample":         ("🎲", "Fraction of rows sampled per tree (row bagging).",
                                  "↑ More rows → less randomness, may overfit outliers.",
                                  "↓ Fewer rows → reduces overfitting via variance reduction.", None),
            "colsample_bytree":  ("🎲", "Fraction of features sampled per tree (col bagging).",
                                  "↑ More features → correlated trees.",
                                  "↓ Fewer features → diverse trees, better generalization.", None),
            "min_samples_split": ("✂️", "Min samples a node needs before it can split.",
                                  "↑ Higher → coarser splits, less overfitting.",
                                  "↓ Lower → finer splits, risk of memorizing noise.", None),
            "min_samples_leaf":  ("🍃", "Min samples required at any leaf node.",
                                  "↑ Higher → smoother, more generalized boundary.",
                                  "↓ Lower → very specific leaves, high overfit risk.", None),
            "max_features":      ("🔀", "Features considered at each split.",
                                  "None → all features used, trees become correlated.",
                                  "'sqrt' → diverse trees — RF default.",
                                  "📌 'sqrt' is the professional default for classification."),
            "criterion":         ("📏", "Split quality measure: Gini vs. Information Gain.",
                                  "Entropy → slightly more informative, slower.",
                                  "Gini → faster, usually equally effective.", None),
            "lr_C":              ("⚖️", "Inverse regularization (smaller = stronger penalty).",
                                  "↑ High C → fits training data closely, overfit risk.",
                                  "↓ Low C → stronger penalty, may underfit.", None),
            "max_iter":          ("🔄", "Max iterations for the solver to converge.",
                                  "↑ More iterations → ensures convergence.",
                                  "↓ Fewer → faster, but may not fully converge.",
                                  "⚠️ Raise this if you see ConvergenceWarning."),
            "solver":            ("⚙️", "Optimization algorithm used to fit the model.",
                                  "'lbfgs' → best for small/medium data.",
                                  "'saga' → supports L1/L2, scales to large datasets.", None),
            "num_leaves":        ("🌿", "Max terminal nodes per tree (key LightGBM knob).",
                                  "↑ More leaves → higher accuracy, rapid overfit risk.",
                                  "↓ Fewer leaves → simpler, safer model.",
                                  "📌 Rule: keep num_leaves ≤ 2^max_depth (31 = 2^5)."),
            "min_child_samples": ("🧑‍🤝‍🧑", "Min observations required to form a leaf node.",
                                  "↑ Higher → conservative splits, ↓ overfitting.",
                                  "↓ Lower → can create leaves for tiny noisy groups.",
                                  "📌 If Train AUC ≫ Test AUC, increase to 50–100."),
            "svm_C":             ("⚖️", "Trade-off: wide margin vs. correct classifications.",
                                  "↑ High C → hard margin, complex boundary, overfit.",
                                  "↓ Low C → soft margin, smoother boundary, underfit.",
                                  "⚠️ SVM is scale-sensitive — data is auto-standardized."),
            "gamma":             ("📡", "How far influence of one training point reaches.",
                                  "↑ High gamma → tight zones, jagged boundary, overfit.",
                                  "↓ Low gamma → broad influence, smooth boundary.",
                                  "📌 'scale' adjusts to data variance — preferred over 'auto'."),
            "n_neighbors":       ("👥", "Number of nearest neighbors voting on each prediction.",
                                  "↑ More neighbors → smoother boundary, less overfit.",
                                  "↓ Fewer neighbors → very local decisions, noisy boundary.",
                                  "⚠️ Prediction time grows with dataset size (lazy learner)."),
            "weights":           ("🏋️", "How much influence each neighbor has in the vote.",
                                  "'distance' → closer neighbors count more.",
                                  "'uniform' → equal vote, smoother boundary.", None),
            "metric":            ("📐", "Formula used to calculate distance between points.",
                                  "'euclidean' → straight-line, sensitive to scale.",
                                  "'manhattan' → taxicab, better in high dimensions.",
                                  "⚠️ KNN/SVM require scaling — applied via Pipeline."),
            "cv_folds":          ("🔁", "Number of cross-validation folds for evaluation.",
                                  "↑ More folds → more reliable estimate, longer evaluation.",
                                  "↓ Fewer folds → faster, but noisier estimate.", None),
        }

        def show_tip(key):
            tip = PARAM_TIPS.get(key)
            if not tip:
                return
            icon, what, up, down, warn = tip
            warn_html = (f"<div style='margin-top:3px;color:#f4a261;font-size:0.77rem;'>{warn}</div>"
                         if warn else "")
            st.markdown(f"""
            <div style='background:rgba(0,180,216,0.07);border-left:3px solid #00b4d8;
                        border-radius:0 8px 8px 0;padding:5px 10px;margin:-4px 0 10px 0;
                        font-size:0.79rem;line-height:1.5;'>
              <span style='font-size:0.95rem;'>{icon}</span>
              <span style='color:#cdd9e5;'> {what}</span><br>
              <span style='color:#06d6a0;'>{up}</span><br>
              <span style='color:#8899aa;'>{down}</span>
              {warn_html}
            </div>""", unsafe_allow_html=True)

        # ── Controls ──────────────────────────────────────────────────────────
        params = {}

        if model_choice == "Logistic Regression":
            params["C"] = st.select_slider("Regularization (C)",
                [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0); show_tip("lr_C")
            params["max_iter"] = st.slider("Max Iterations", 100, 2000, 500, 100); show_tip("max_iter")
            params["solver"] = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"]); show_tip("solver")

        elif model_choice == "Decision Tree":
            params["max_depth"] = st.slider("Max Depth", 2, 20, 8); show_tip("max_depth")
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 5); show_tip("min_samples_split")
            params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 10, 2); show_tip("min_samples_leaf")
            params["criterion"] = st.selectbox("Criterion", ["gini", "entropy"]); show_tip("criterion")

        elif model_choice == "Random Forest":
            params["n_estimators"] = st.slider("Number of Trees", 50, 500, 100, 50); show_tip("n_estimators")
            params["max_depth"] = st.slider("Max Depth", 2, 20, 10); show_tip("max_depth")
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 5); show_tip("min_samples_split")
            params["max_features"] = st.selectbox("Max Features", ["sqrt", "log2", None]); show_tip("max_features")

        elif model_choice == "Gradient Boosting":
            params["n_estimators"] = st.slider("Number of Trees", 50, 400, 100, 50); show_tip("n_estimators")
            params["learning_rate"] = st.select_slider("Learning Rate",
                [0.01, 0.05, 0.1, 0.15, 0.2, 0.3], value=0.1); show_tip("learning_rate")
            params["max_depth"] = st.slider("Max Depth", 2, 8, 4); show_tip("max_depth")
            params["subsample"] = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05); show_tip("subsample")

        elif model_choice == "XGBoost":
            params["n_estimators"] = st.slider("Number of Trees", 50, 400, 100, 50); show_tip("n_estimators")
            params["learning_rate"] = st.select_slider("Learning Rate",
                [0.01, 0.05, 0.1, 0.15, 0.2, 0.3], value=0.1); show_tip("learning_rate")
            params["max_depth"] = st.slider("Max Depth", 2, 10, 5); show_tip("max_depth")
            params["subsample"] = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05); show_tip("subsample")
            params["colsample_bytree"] = st.slider("Column Sample/Tree", 0.5, 1.0, 0.8, 0.05); show_tip("colsample_bytree")

        elif model_choice == "LightGBM":
            params["n_estimators"] = st.slider("Number of Trees", 50, 400, 100, 50); show_tip("n_estimators")
            params["learning_rate"] = st.select_slider("Learning Rate",
                [0.01, 0.05, 0.1, 0.15, 0.2, 0.3], value=0.1); show_tip("learning_rate")
            params["max_depth"] = st.slider("Max Depth", 2, 10, 5); show_tip("max_depth")
            params["num_leaves"] = st.slider("Num Leaves", 10, 100, 31, 5); show_tip("num_leaves")
            params["min_child_samples"] = st.slider("Min Child Samples", 5, 50, 20, 5); show_tip("min_child_samples")

        elif model_choice == "K-Nearest Neighbors":
            params["n_neighbors"] = st.slider("Number of Neighbors (K)", 1, 31, 7, 2); show_tip("n_neighbors")
            params["weights"] = st.selectbox("Weights", ["uniform", "distance"]); show_tip("weights")
            params["metric"] = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"]); show_tip("metric")

        elif model_choice == "SVM (RBF)":
            params["C"] = st.select_slider("Regularization (C)",
                [0.01, 0.1, 1.0, 5.0, 10.0, 50.0], value=1.0); show_tip("svm_C")
            params["gamma"] = st.selectbox("Gamma", ["scale", "auto"]); show_tip("gamma")

        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5); show_tip("cv_folds")
        train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

    with col_right:
        if train_btn:
            with st.spinner(f"Training {model_choice}..."):
                # Build classifier with custom params
                if model_choice == "Logistic Regression":
                    clf_ht = Pipeline([("scaler", StandardScaler()),
                                       ("clf", LogisticRegression(
                                           C=params["C"], max_iter=params["max_iter"],
                                           solver=params["solver"], random_state=RANDOM_STATE,
                                           class_weight="balanced"))])
                elif model_choice == "Decision Tree":
                    clf_ht = DecisionTreeClassifier(
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        criterion=params["criterion"],
                        random_state=RANDOM_STATE, class_weight="balanced")
                elif model_choice == "Random Forest":
                    clf_ht = RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        max_features=params["max_features"],
                        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
                elif model_choice == "Gradient Boosting":
                    clf_ht = GradientBoostingClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        subsample=params["subsample"],
                        random_state=RANDOM_STATE)
                elif model_choice == "XGBoost":
                    clf_ht = XGBClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        scale_pos_weight=scale_pos_wt,
                        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
                elif model_choice == "LightGBM":
                    clf_ht = LGBMClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        num_leaves=params["num_leaves"],
                        min_child_samples=params["min_child_samples"],
                        class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
                elif model_choice == "K-Nearest Neighbors":
                    clf_ht = Pipeline([("scaler", StandardScaler()),
                                       ("clf", KNeighborsClassifier(
                                           n_neighbors=params["n_neighbors"],
                                           weights=params["weights"],
                                           metric=params["metric"], n_jobs=-1))])
                elif model_choice == "SVM (RBF)":
                    clf_ht = Pipeline([("scaler", StandardScaler()),
                                       ("clf", SVC(kernel="rbf", C=params["C"],
                                                   gamma=params["gamma"], probability=True,
                                                   class_weight="balanced",
                                                   random_state=RANDOM_STATE))])

                skf_ht = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                          random_state=RANDOM_STATE)

                f1_cv   = cross_val_score(clf_ht, X, y, cv=skf_ht, scoring=scoring_f1,   n_jobs=-1)
                auc_cv  = cross_val_score(clf_ht, X, y, cv=skf_ht, scoring="roc_auc",    n_jobs=-1)
                acc_cv  = cross_val_score(clf_ht, X, y, cv=skf_ht, scoring="accuracy",   n_jobs=-1)
                prec_cv = cross_val_score(clf_ht, X, y, cv=skf_ht, scoring=scoring_prec, n_jobs=-1)
                rec_cv  = cross_val_score(clf_ht, X, y, cv=skf_ht, scoring=scoring_rec,  n_jobs=-1)

                clf_ht.fit(X, y)
                y_pred_ht = clf_ht.predict(X)
                y_prob_ht = (clf_ht.predict_proba(X)[:, 1]
                             if hasattr(clf_ht, "predict_proba") else None)
                cm_ht = confusion_matrix(y, y_pred_ht)

            # KPI summary
            k1, k2, k3, k4, k5 = st.columns(5)
            for kc, kl, kv in [(k1, "F1",        f1_cv.mean()),
                                 (k2, "AUC-ROC",   auc_cv.mean()),
                                 (k3, "Accuracy",  acc_cv.mean()),
                                 (k4, "Precision", prec_cv.mean()),
                                 (k5, "Recall",    rec_cv.mean())]:
                with kc:
                    st.markdown(f"""
                    <div class='metric-card' style='padding:0.7rem;'>
                        <div class='metric-label'>{kl}</div>
                        <div class='metric-value' style='font-size:1.6rem;'>{kv:.4f}</div>
                        <div class='metric-delta'>±{[f1_cv,auc_cv,acc_cv,prec_cv,rec_cv][[k1,k2,k3,k4,k5].index(kc)].std():.4f}</div>
                    </div>""", unsafe_allow_html=True)

            col_fold, col_cm_ht = st.columns([1, 1])

            with col_fold:
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(cv_folds)],
                    "F1": f1_cv, "AUC": auc_cv, "Accuracy": acc_cv,
                })
                fig_fold = go.Figure()
                for metric, color in [("F1", COLORS["primary"]),
                                       ("AUC", COLORS["success"]),
                                       ("Accuracy", COLORS["warning"])]:
                    fig_fold.add_trace(go.Bar(
                        name=metric, x=fold_df["Fold"], y=fold_df[metric],
                        marker_color=color, opacity=0.85,
                    ))
                fig_fold.update_layout(
                    barmode="group",
                    title=dict(text="CV Fold Performance", font=dict(size=14, color="#00b4d8")),
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
                    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                    xaxis=dict(gridcolor="#1a2a3a", linecolor="#2a3a4a"),
                    yaxis=dict(range=[0, 1.1], gridcolor="#1a2a3a", linecolor="#2a3a4a"),
                    height=320, margin=dict(t=50, b=10),
                    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
                )
                st.plotly_chart(fig_fold, use_container_width=True)

            with col_cm_ht:
                pct_ht = cm_ht.astype(float) / cm_ht.sum(axis=1, keepdims=True) * 100
                ann_ht = [[f"{v}<br>({p:.1f}%)" for v, p in zip(rv, rp)]
                           for rv, rp in zip(cm_ht, pct_ht)]
                fig_cm_ht = go.Figure(go.Heatmap(
                    z=cm_ht, x=[f"Pred: {c}" for c in le.classes_],
                    y=[f"Actual: {c}" for c in le.classes_],
                    text=ann_ht, texttemplate="%{text}",
                    textfont=dict(size=13), colorscale="YlOrRd", showscale=True,
                ))
                fig_cm_ht.update_layout(
                    title=dict(text=f"Confusion Matrix — {model_choice}",
                               font=dict(size=14, color="#00b4d8")),
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
                    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                    xaxis=dict(gridcolor="#1a2a3a"), yaxis=dict(gridcolor="#1a2a3a"),
                    height=320, margin=dict(t=50, b=30),
                )
                st.plotly_chart(fig_cm_ht, use_container_width=True)

            # ROC curve
            if y_prob_ht is not None:
                fpr_ht, tpr_ht, _ = roc_curve(y, y_prob_ht)
                fig_roc_ht = go.Figure()
                fig_roc_ht.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="#adb5bd", dash="dash", width=1.5), name="Random",
                ))
                fig_roc_ht.add_trace(go.Scatter(
                    x=fpr_ht, y=tpr_ht, mode="lines",
                    line=dict(color=COLORS["primary"], width=3),
                    fill="tozeroy", fillcolor="rgba(0,180,216,0.1)",
                    name=f"{model_choice} (AUC={auc_cv.mean():.4f})",
                ))
                fig_roc_ht.update_layout(
                    title=dict(text="ROC Curve (Training Set)", font=dict(size=14, color="#00b4d8")),
                    xaxis=dict(title="False Positive Rate", gridcolor="#1a2a3a"),
                    yaxis=dict(title="True Positive Rate", gridcolor="#1a2a3a"),
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
                    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                    height=320, margin=dict(t=50, b=10),
                    legend=dict(x=0.5, y=0.1, bgcolor="rgba(0,0,0,0.5)"),
                )
                st.plotly_chart(fig_roc_ht, use_container_width=True)

            # Feature importance
            est_ht = (clf_ht.named_steps.get("clf", clf_ht)
                      if hasattr(clf_ht, "named_steps") else clf_ht)
            if hasattr(est_ht, "feature_importances_"):
                fi_ht = pd.Series(est_ht.feature_importances_, index=X.columns)
                fi_ht = fi_ht.sort_values(ascending=False).head(15)
                fig_fi = go.Figure(go.Bar(
                    y=fi_ht.index[::-1], x=fi_ht.values[::-1],
                    orientation="h",
                    marker=dict(color=fi_ht.values[::-1],
                                colorscale="Plasma", showscale=True),
                    text=fi_ht.values[::-1].round(4),
                    textposition="outside", textfont=dict(color="white", size=10),
                ))
                fig_fi.update_layout(
                    title=dict(text="Top 15 Feature Importances",
                               font=dict(size=14, color="#00b4d8")),
                    xaxis=dict(title="Importance", gridcolor="#1a2a3a"),
                    yaxis=dict(gridcolor="#1a2a3a"),
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#111d2d",
                    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                    height=420, margin=dict(t=50, b=10, r=80),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;
                        background:rgba(0,180,216,0.05);border-radius:12px;
                        border:1px dashed #00b4d8;margin-top:1rem;'>
                <span style='font-size:4rem;'>⚙️</span>
                <h3 style='color:#00b4d8;margin:0.5rem 0;'>Configure & Train</h3>
                <p style='color:#adb5bd;'>
                    Select a model and adjust hyperparameters on the left,<br>
                    then click <strong style='color:#06d6a0;'>🚀 Train Model</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#4a6374;font-size:0.8rem;padding:0.5rem;'>
    🏦 Bank Client Churn Dashboard &nbsp;|&nbsp;
    5-Fold Stratified Cross-Validation &nbsp;|&nbsp;
    DV: Status (Attrited / Existing) &nbsp;|&nbsp;
    19 features per Data Dictionary
</div>
""", unsafe_allow_html=True)
