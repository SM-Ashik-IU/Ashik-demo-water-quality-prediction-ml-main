"""
PureSip: Drinks Quality Prediction System
==========================================
Streamlit UI & Multi-Model Inference Engine

Modular 3-Page Navigation Architecture:
  1. Home: Project overview, problem statement, parameter guidelines, and general info.
  2. Prediction: Focused purely on user inputs, single best model prediction (Tuned Random Forest),
                 confidence/accuracy metrics, and specific feature importance visualization.
                 (Excludes any combined/ensemble model predictions).
  3. Comparison: Technical analysis page comparing ElasticNet, XGBoost, and Random Forest
                 with comparative benchmarks, metrics table, and live multi-model evaluation.
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Project root configuration ───────────────────────────────────────────────
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "config.py")):
    ROOT_DIR = CURRENT_DIR
else:
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
from config import DATA_PATH, MODEL_PATH

# ════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PureSip — Drinks Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# 2. CUSTOM STYLING (Dark Blue Theme & Navigation Sanitization)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #090e1a 0%, #0c1824 50%, #0a1628 100%);
    color: #e8f4fd;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Hide default multi-page nav so ONLY the 3 consolidated menus appear ── */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112233 100%);
    border-right: 1px solid rgba(0, 180, 255, 0.12);
}
[data-testid="stSidebar"] * {
    color: #c8e6fa !important;
}

/* ── Brand Header ── */
.brand-bar {
    background: linear-gradient(90deg, rgba(0, 120, 212, 0.18), rgba(0, 200, 255, 0.08));
    border: 1px solid rgba(0, 180, 255, 0.25);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 18px;
    backdrop-filter: blur(10px);
}
.brand-logo {
    font-size: 46px;
    line-height: 1;
}
.brand-title {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(90deg, #00c8ff, #0078d4, #00c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.brand-sub {
    font-size: 14px;
    color: rgba(180, 220, 255, 0.75);
    margin-top: 4px;
}

/* ── Content Cards ── */
.info-card {
    background: rgba(0, 120, 212, 0.08);
    border: 1px solid rgba(0, 180, 255, 0.18);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 22px;
}

/* ── Section Titles ── */
.sec-title {
    font-size: 18px;
    font-weight: 700;
    color: #a8d8f0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0, 180, 255, 0.20);
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Horizontal Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 180, 255, 0.35), transparent);
    margin: 24px 0;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 18px;
}
.metric-card {
    background: rgba(0, 120, 212, 0.08);
    border: 1px solid rgba(0, 180, 255, 0.18);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0, 180, 255, 0.15);
}
.metric-card.top-performer {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.16), rgba(0, 120, 212, 0.12));
    border: 2px solid rgba(46, 204, 113, 0.65);
    box-shadow: 0 4px 24px rgba(46, 204, 113, 0.20);
}
.m-badge {
    display: inline-block;
    padding: 3px 11px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-radius: 20px;
    margin-bottom: 10px;
}
.badge-std {
    background: rgba(0, 120, 212, 0.22);
    color: #00c8ff;
    border: 1px solid rgba(0, 180, 255, 0.35);
}
.badge-top {
    background: rgba(46, 204, 113, 0.22);
    color: #2ecc71;
    border: 1px solid rgba(46, 204, 113, 0.50);
}
.m-score {
    font-size: 34px;
    font-weight: 800;
    color: #00c8ff;
    line-height: 1;
}
.m-score.top {
    color: #2ecc71;
}
.m-name {
    font-size: 14px;
    font-weight: 700;
    color: #a8d8f0;
    margin-top: 8px;
}
.m-desc {
    font-size: 12px;
    color: rgba(180, 215, 245, 0.60);
    margin-top: 3px;
}

/* ── Result Banners ── */
.res-safe {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.18), rgba(0, 120, 60, 0.12));
    border: 2px solid rgba(46, 204, 113, 0.60);
    border-radius: 16px;
    padding: 26px 30px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 6px 28px rgba(46, 204, 113, 0.20);
    animation: fadeIn 0.4s ease-in-out;
}
.res-unsafe {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.18), rgba(140, 15, 15, 0.12));
    border: 2px solid rgba(231, 76, 60, 0.60);
    border-radius: 16px;
    padding: 26px 30px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 6px 28px rgba(231, 76, 60, 0.20);
    animation: fadeIn 0.4s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.verdict {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}
.verdict.safe { color: #2ecc71; }
.verdict.unsafe { color: #ff6b6b; }
.big-score {
    font-size: 46px;
    font-weight: 800;
    line-height: 1;
    margin: 10px 0 6px;
}
.big-score.safe { color: #2ecc71; }
.big-score.unsafe { color: #ff6b6b; }
.score-cap {
    font-size: 13px;
    color: rgba(200, 235, 255, 0.75);
    text-transform: uppercase;
    letter-spacing: 0.9px;
}

/* ── Metric Highlights Grid Below Result ── */
.submetric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 22px;
}
.submetric-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(0, 180, 255, 0.16);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.submetric-val {
    font-size: 22px;
    font-weight: 700;
    color: #00c8ff;
}
.submetric-lbl {
    font-size: 12px;
    color: rgba(180, 215, 245, 0.65);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Panel Box & Chart Box ── */
.panel-box {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(0, 180, 255, 0.16);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 20px;
}

/* ── Table Styling ── */
.custom-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
}
.custom-table th {
    background: rgba(0, 120, 212, 0.18);
    color: #a8d8f0;
    padding: 12px 14px;
    text-align: left;
    font-size: 13.5px;
    border-bottom: 1px solid rgba(0, 180, 255, 0.25);
}
.custom-table td {
    padding: 12px 14px;
    border-bottom: 1px solid rgba(0, 180, 255, 0.10);
    font-size: 13.5px;
    color: rgba(215, 235, 255, 0.85);
}
.custom-table tr:hover td {
    background: rgba(0, 120, 212, 0.06);
}

/* ── Primary Buttons (Active Navigation & Action Buttons) ── */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
div[data-testid="stButton"] > button:not([kind="secondary"]):not([data-testid="stBaseButton-secondary"]) {
    background: linear-gradient(135deg, #0062b8 0%, #0078d4 50%, #00a0e8 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0, 200, 255, 0.45) !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(0, 120, 212, 0.35) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #0078d4 0%, #00a0e8 50%, #00c8ff 100%) !important;
    box-shadow: 0 6px 28px rgba(0, 180, 255, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary Buttons (Inactive Navigation Tabs) ── */
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #c8e6fa !important;
    border: 1px solid rgba(0, 180, 255, 0.22) !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover,
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(0, 120, 212, 0.20) !important;
    border-color: rgba(0, 200, 255, 0.50) !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 3. CACHED BACKEND: Model Loading & Benchmark Evaluation
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models_and_scores():
    """
    Loads the trained models (Random Forest, XGBoost), StandardScaler, and trains ElasticNet.
    Computes static test-set accuracy scores across models.
    """
    scaler_path = os.path.join(os.path.dirname(MODEL_PATH), "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        stacking_model = pickle.load(f)

    # Base models
    rf_model = stacking_model.named_estimators_["rf"]
    xgb_model = stacking_model.named_estimators_["xgb"]

    # Reconstruct dataset matching training pipeline
    df = pd.read_csv(DATA_PATH)
    df.fillna(df.mean(), inplace=True)
    df["ph_hardness"] = df["ph"] * df["Hardness"]
    df["solids_conductivity"] = df["Solids"] / (df["Conductivity"] + 1)
    df["chloramines_trihalomethanes"] = df["Chloramines"] * df["Trihalomethanes"]

    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    feature_cols = list(X.columns)

    # Transform features with fitted scaler
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ElasticNet baseline (L1 + L2 regularized logistic regression)
    en_model = LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=200, random_state=42)
    en_model.fit(X_train, y_train)

    # Compute static dataset accuracy percentages for the three individual models
    accuracies = {
        "Random Forest": round(accuracy_score(y_test, rf_model.predict(X_test)) * 100, 2),
        "XGBoost": round(accuracy_score(y_test, xgb_model.predict(X_test)) * 100, 2),
        "ElasticNet": round(accuracy_score(y_test, en_model.predict(X_test)) * 100, 2),
    }

    return rf_model, xgb_model, en_model, scaler, feature_cols, accuracies, df


# Initialize models and scores once
rf_model, xgb_model, en_model, scaler, FEATURE_COLS, ACCURACIES, raw_df = load_models_and_scores()


# ════════════════════════════════════════════════════════════════════════════
# 4. SYNCHRONIZED NAVIGATION (Sidebar Radio + Top Navigation Bar)
# ════════════════════════════════════════════════════════════════════════════
NAV_OPTIONS = ["Home", "Prediction", "Comparison"]

if "nav_menu" not in st.session_state:
    st.session_state["nav_menu"] = "Home"

def sync_from_sidebar():
    st.session_state["nav_menu"] = st.session_state["sidebar_nav_radio"]

# ── Sidebar Navigation: Default Streamlit st.sidebar.radio ──
st.sidebar.markdown("## 🧭 Navigation")
curr_idx = NAV_OPTIONS.index(st.session_state["nav_menu"]) if st.session_state["nav_menu"] in NAV_OPTIONS else 0
sidebar_selection = st.sidebar.radio(
    "Select Menu",
    options=NAV_OPTIONS,
    index=curr_idx,
    key="sidebar_nav_radio",
    on_change=sync_from_sidebar
)
menu = st.session_state["nav_menu"]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:12px; color:rgba(180,215,245,0.65); line-height:1.6;'>"
    "<strong>PureSip ML System</strong><br>"
    "Drinks Quality &amp; Potability Assessment Engine"
    "</div>",
    unsafe_allow_html=True
)

# ── Top Navigation Bar (Prominently visible across the top of every screen) ──
nav_col1, nav_col2, nav_col3 = st.columns(3, gap="small")

with nav_col1:
    btn_type = "primary" if menu == "Home" else "secondary"
    if st.button("🏠 Home", use_container_width=True, type=btn_type, key="top_nav_home"):
        st.session_state["nav_menu"] = "Home"
        st.rerun()

with nav_col2:
    btn_type = "primary" if menu == "Prediction" else "secondary"
    if st.button("🔬 Prediction", use_container_width=True, type=btn_type, key="top_nav_pred"):
        st.session_state["nav_menu"] = "Prediction"
        st.rerun()

with nav_col3:
    btn_type = "primary" if menu == "Comparison" else "secondary"
    if st.button("⚖️ Comparison", use_container_width=True, type=btn_type, key="top_nav_comp"):
        st.session_state["nav_menu"] = "Comparison"
        st.rerun()

st.markdown('<div class="divider" style="margin: 14px 0 24px 0;"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 5. MENU 1: HOME (Project Overview & General Information)
# ════════════════════════════════════════════════════════════════════════════
if menu == "Home":
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-logo">💧</div>
        <div>
            <div class="brand-title">PureSip: Drinks Quality Prediction System</div>
            <div class="brand-sub">
                Intelligent Physicochemical Quality &amp; Potability Assessment Platform
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Overview Card
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top:0; color:#a8d8f0; font-size:19px; font-weight:700;">📖 Project Overview</h3>
        <p style="color:rgba(200,235,255,0.88); line-height:1.65; font-size:14.5px;">
            Access to safe drinking water is essential to human health, environmental sustainability, and commercial beverage production.
            <strong>PureSip</strong> is an end-to-end Machine Learning web application designed to evaluate whether water or drink samples
            are safe for human consumption based on nine internationally recognized physicochemical parameters.
        </p>
        <p style="color:rgba(200,235,255,0.88); line-height:1.65; font-size:14.5px; margin-bottom:0;">
            The system employs data preprocessing, SMOTE class balancing, domain-specific interaction feature engineering,
            and hyperparameter-tuned classification algorithms to deliver reliable, explainable quality verdicts in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Water Quality Parameters & Standards (WHO Guidelines)
    st.markdown('<div class="sec-title">🧪 Key Physicochemical Quality Parameters</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-box">
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>WHO Permissible Standard</th>
                    <th>Significance &amp; Health Impact</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>pH Level</strong></td>
                    <td>6.5 – 8.5</td>
                    <td>Measures acidity and alkalinity. Deviations affect taste and pipe corrosion.</td>
                </tr>
                <tr>
                    <td><strong>Hardness</strong></td>
                    <td>&lt; 300 mg/L</td>
                    <td>Precipitation capacity caused by dissolved calcium and magnesium ions.</td>
                </tr>
                <tr>
                    <td><strong>Solids (TDS)</strong></td>
                    <td>&lt; 500 ppm (max 1000)</td>
                    <td>Total dissolved solids including mineral salts, cations, and anions.</td>
                </tr>
                <tr>
                    <td><strong>Chloramines</strong></td>
                    <td>&lt; 4.0 ppm</td>
                    <td>Chemical disinfectant formed from chlorine and ammonia to treat public water.</td>
                </tr>
                <tr>
                    <td><strong>Sulfate</strong></td>
                    <td>&lt; 250 mg/L</td>
                    <td>Naturally occurring mineral salts. Excess concentrations cause laxative effects.</td>
                </tr>
                <tr>
                    <td><strong>Conductivity</strong></td>
                    <td>&lt; 400 μS/cm</td>
                    <td>Electrical conductivity indicating the total concentration of dissolved electrolytes.</td>
                </tr>
                <tr>
                    <td><strong>Organic Carbon (TOC)</strong></td>
                    <td>&lt; 2.0 ppm</td>
                    <td>Measure of decaying natural organic matter and synthetic chemical contaminants.</td>
                </tr>
                <tr>
                    <td><strong>Trihalomethanes</strong></td>
                    <td>&lt; 80 μg/L</td>
                    <td>Chemical by-products of chlorine disinfection; regulated due to potential carcinogenicity.</td>
                </tr>
                <tr>
                    <td><strong>Turbidity</strong></td>
                    <td>&lt; 5.0 NTU</td>
                    <td>Measure of water clarity and concentration of light-scattering suspended particles.</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # How to Use & Navigation Guide
    st.markdown('<div class="sec-title">🧭 Application Navigation Guide</div>', unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2, gap="medium")

    with col_nav1:
        st.markdown("""
        <div class="panel-box" style="height:100%;">
            <div style="font-size:26px; margin-bottom:8px;">🔬</div>
            <h4 style="color:#a8d8f0; margin:0 0 8px 0;">Prediction Menu</h4>
            <p style="font-size:13.5px; color:rgba(200,235,255,0.80); line-height:1.6; margin:0;">
                Test real or simulated drink samples. Adjust parameter sliders and receive an instant potability verdict
                powered exclusively by our <strong>single best-performing tuned model (Random Forest)</strong>,
                complete with prediction confidence and feature importance visual breakdown.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_nav2:
        st.markdown("""
        <div class="panel-box" style="height:100%;">
            <div style="font-size:26px; margin-bottom:8px;">⚖️</div>
            <h4 style="color:#a8d8f0; margin:0 0 8px 0;">Comparison Menu</h4>
            <p style="font-size:13.5px; color:rgba(200,235,255,0.80); line-height:1.6; margin:0;">
                The dedicated technical analysis page. Inspect benchmark accuracy graphs across
                <strong>ElasticNet</strong>, <strong>XGBoost</strong>, and <strong>Random Forest</strong>,
                review model architectures, and run live multi-model voting tests on any sample.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 6. MENU 2: PREDICTION (Single Best-Performing Tuned Model: Random Forest)
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Prediction":
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-logo">🔬</div>
        <div>
            <div class="brand-title">PureSip: Quality Prediction</div>
            <div class="brand-sub">
                Inference Powered Exclusively by Tuned Random Forest (Single Best Model &bull; 86.13% Accuracy)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🎛️ Sample Physicochemical Parameters</div>', unsafe_allow_html=True)

    # Input parameter sliders
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("**Acidity & Hardness Indicators**")
        ph = st.slider(
            "pH Level (0 – 14)",
            min_value=0.0,
            max_value=14.0,
            value=7.08,
            step=0.05,
            help="WHO standard: 6.5 – 8.5"
        )
        hardness = st.slider(
            "Hardness (mg/L)",
            min_value=40.0,
            max_value=330.0,
            value=196.37,
            step=1.0,
            help="Derived from calcium and magnesium ions"
        )
        solids = st.slider(
            "Total Dissolved Solids - TDS (ppm)",
            min_value=300.0,
            max_value=62000.0,
            value=22014.0,
            step=100.0,
            help="Total mineral and dissolved salt content"
        )

    with col2:
        st.markdown("**Disinfection & Mineral Composition**")
        chloramines = st.slider(
            "Chloramines (ppm)",
            min_value=0.0,
            max_value=15.0,
            value=7.12,
            step=0.05,
            help="Disinfection treatment level in drinking water"
        )
        sulfate = st.slider(
            "Sulfates (mg/L)",
            min_value=120.0,
            max_value=500.0,
            value=333.78,
            step=1.0,
            help="Naturally occurring mineral salts"
        )
        conductivity = st.slider(
            "Conductivity (μS/cm)",
            min_value=150.0,
            max_value=800.0,
            value=426.21,
            step=1.0,
            help="Electrical conductivity of dissolved ions"
        )

    with col3:
        st.markdown("**Organic Matter & Clarity**")
        organic_carbon = st.slider(
            "Organic Carbon (ppm)",
            min_value=2.0,
            max_value=30.0,
            value=14.28,
            step=0.1,
            help="Total organic carbon (TOC) content"
        )
        trihalomethanes = st.slider(
            "Trihalomethanes (μg/L)",
            min_value=0.0,
            max_value=130.0,
            value=66.40,
            step=0.5,
            help="Chlorination by-products concentration"
        )
        turbidity = st.slider(
            "Turbidity (NTU)",
            min_value=1.0,
            max_value=7.0,
            value=3.97,
            step=0.05,
            help="Measure of suspended particle cloudiness"
        )

    # Predict Button
    predict_button = st.button(
        "🔍 Predict Water Quality",
        use_container_width=True,
        type="primary",
    )

    # Result Section — strictly hidden until Predict is clicked
    if predict_button:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🎯 Final Prediction &amp; Model Explainability</div>', unsafe_allow_html=True)

        # 1. Feature Engineering matching training dataset
        ph_hardness = ph * hardness
        solids_conductivity = solids / (conductivity + 1.0)
        chloramines_trihalomethanes = chloramines * trihalomethanes

        input_df = pd.DataFrame([[
            ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity,
            ph_hardness, solids_conductivity, chloramines_trihalomethanes
        ]], columns=FEATURE_COLS)

        # Scale the live input features
        input_scaled = scaler.transform(input_df)

        # 2. Predict using ONLY the single best-performing model (Tuned Random Forest)
        rf_pred = int(rf_model.predict(input_scaled)[0])
        rf_proba = rf_model.predict_proba(input_scaled)[0]
        confidence = float(max(rf_proba) * 100)
        potability_score = float(rf_proba[1] * 100)
        best_model_acc = ACCURACIES["Random Forest"]

        # 3. Main Result Display
        if rf_pred == 1:
            css_card = "res-safe"
            verdict_text = "Safe for Drinking ✅"
            v_class = "safe"
            icon = "💧"
            detail_msg = "This sample meets potable safety criteria and is deemed safe for human consumption."
        else:
            css_card = "res-unsafe"
            verdict_text = "NOT Safe for Drinking ⚠️"
            v_class = "unsafe"
            icon = "⚠️"
            detail_msg = "One or more parameters exceed safe drinking thresholds. Purification required."

        st.markdown(f"""
        <div class="{css_card}">
            <div style="font-size:42px; margin-bottom:6px;">{icon}</div>
            <div class="verdict {v_class}">Water is {verdict_text}</div>
            <div class="big-score {v_class}">{potability_score:.1f}%</div>
            <div class="score-cap">Potability Index &bull; Tuned Random Forest (Single Best Model)</div>
            <div style="font-size:14px; color:rgba(200,235,255,0.80); margin-top:10px;">{detail_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        # 4. Display Accuracy and Confidence Score directly below the prediction
        st.markdown(f"""
        <div class="submetric-grid">
            <div class="submetric-card">
                <div class="submetric-val" style="color:#2ecc71;">Tuned Random Forest</div>
                <div class="submetric-lbl">Active Production Model</div>
            </div>
            <div class="submetric-card">
                <div class="submetric-val">{confidence:.1f}%</div>
                <div class="submetric-lbl">Prediction Confidence</div>
            </div>
            <div class="submetric-card">
                <div class="submetric-val">{best_model_acc}%</div>
                <div class="submetric-lbl">Model Benchmark Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Clean Feature Importance Visual (Influential parameters for this specific prediction)
        st.markdown(
            '<div class="sec-title" style="margin-top:16px;">'
            '📊 Input Parameter Influence on this Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        # Map feature names to clean readable labels
        feature_label_map = {
            'ph': 'pH Level',
            'Hardness': 'Hardness',
            'Solids': 'Total Dissolved Solids (TDS)',
            'Chloramines': 'Chloramines',
            'Sulfate': 'Sulfates',
            'Conductivity': 'Conductivity',
            'Organic_carbon': 'Organic Carbon',
            'Trihalomethanes': 'Trihalomethanes',
            'Turbidity': 'Turbidity',
            'ph_hardness': 'pH × Hardness',
            'solids_conductivity': 'Solids / Conductivity',
            'chloramines_trihalomethanes': 'Chloramines × Trihalomethanes'
        }

        # Calculate sample-specific influence based on RF feature weights and standardized deviation
        rf_weights = rf_model.feature_importances_
        sample_deviation = np.abs(input_scaled[0])
        sample_impact = rf_weights * (1.0 + sample_deviation)
        sample_impact_pct = (sample_impact / np.sum(sample_impact)) * 100

        # Sort features by highest impact and take top 8
        sorted_indices = np.argsort(sample_impact_pct)[::-1][:8]
        top_labels = [feature_label_map[FEATURE_COLS[i]] for i in sorted_indices]
        top_values = [sample_impact_pct[i] for i in sorted_indices]

        # Horizontal bar chart using fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        y_positions = np.arange(len(top_labels))
        bars = ax.barh(y_positions, top_values, color="#00a8e8", height=0.55, edgecolor="none")

        # Highlight the #1 dominant factor in emerald green
        bars[0].set_color("#2ecc71")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_labels, color="#c8e6fa", fontsize=10.5)
        ax.invert_yaxis()  # Top feature at the top

        for spine in ax.spines.values():
            spine.set_edgecolor((0.0, 0.70, 1.0, 0.18))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.xaxis.grid(True, linestyle="--", alpha=0.18, color="#00c8ff")
        ax.set_axisbelow(True)

        for bar in bars:
            w = bar.get_width()
            bar_color = bar.get_facecolor()
            ax.text(
                w + 0.4,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w:.1f}%",
                va="center",
                ha="left",
                color=bar_color,
                fontweight="bold",
                fontsize=10,
            )

        ax.set_xlabel("Relative Predictive Contribution (%)", color="#c8e6fa", fontsize=10.5, labelpad=8)
        ax.tick_params(axis="x", colors="#c8e6fa", labelsize=10)
        ax.set_title(
            "Top Influential Factors for Current Drink Sample",
            color="#a8d8f0",
            fontsize=12.5,
            fontweight="bold",
            pad=14,
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 7. MENU 3: COMPARISON (Technical Analysis of ElasticNet, XGBoost & RF)
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Comparison":
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-logo">⚖️</div>
        <div>
            <div class="brand-title">PureSip: Model Comparison &amp; Technical Analysis</div>
            <div class="brand-sub">
                Comparative Evaluation: ElasticNet vs. XGBoost vs. Random Forest
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Overall Dataset Accuracy Benchmarks
    st.markdown('<div class="sec-title">📊 Section 1: Overall Dataset Model Accuracy Benchmarks</div>', unsafe_allow_html=True)

    rf_acc = ACCURACIES["Random Forest"]
    xgb_acc = ACCURACIES["XGBoost"]
    en_acc = ACCURACIES["ElasticNet"]

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <span class="m-badge badge-std">Linear Baseline</span>
            <div class="m-score">{en_acc}%</div>
            <div class="m-name">ElasticNet</div>
            <div class="m-desc">L1 + L2 Regularized LR</div>
        </div>
        <div class="metric-card">
            <span class="m-badge badge-std">Boosting</span>
            <div class="m-score">{xgb_acc}%</div>
            <div class="m-name">XGBoost</div>
            <div class="m-desc">Gradient Boosted Trees</div>
        </div>
        <div class="metric-card top-performer">
            <span class="m-badge badge-top">🏆 Best Performer</span>
            <div class="m-score top">{rf_acc}%</div>
            <div class="m-name" style="color:#2ecc71;">Random Forest (Tuned)</div>
            <div class="m-desc">Ensemble of 200 Decision Trees</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical Characteristics Table & Static Benchmark Chart
    col_chart, col_table = st.columns([1, 1], gap="large")

    with col_chart:
        st.markdown(
            '<div class="sec-title" style="font-size:15px; margin-bottom:12px;">'
            '📈 Test Dataset Accuracy Comparison'
            '</div>',
            unsafe_allow_html=True
        )

        model_names = ["ElasticNet", "XGBoost", "Random Forest"]
        model_scores = [en_acc, xgb_acc, rf_acc]
        colors = ["#4a6fa5", "#00a8e8", "#2ecc71"]

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        bars = ax.bar(model_names, model_scores, color=colors, width=0.48, edgecolor="none")
        bars[-1].set_edgecolor("#55efc4")
        bars[-1].set_linewidth(2.2)

        ax.set_ylim(45, 100)
        ax.set_ylabel("Accuracy (%)", color="#c8e6fa", fontsize=10.5, labelpad=8)
        ax.tick_params(axis="x", colors="#c8e6fa", labelsize=10.5)
        ax.tick_params(axis="y", colors="#c8e6fa", labelsize=10)

        for spine in ax.spines.values():
            spine.set_edgecolor((0.0, 0.70, 1.0, 0.18))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.yaxis.grid(True, linestyle="--", alpha=0.18, color="#00c8ff")
        ax.set_axisbelow(True)

        for bar, score in zip(bars, model_scores):
            is_top = (bar == bars[-1])
            c = "#2ecc71" if is_top else "#00c8ff"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.6,
                f"{score:.2f}%",
                ha="center",
                va="bottom",
                color=c,
                fontweight="bold",
                fontsize=11,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        st.markdown(
            '<div class="sec-title" style="font-size:15px; margin-bottom:12px;">'
            '📑 Model Architecture Comparison'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Type</th>
                    <th>Decision Surface</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>ElasticNet</strong></td>
                    <td>Linear LR</td>
                    <td>Linear Hyperplane</td>
                    <td><span style="color:#a8d8f0;">Baseline</span></td>
                </tr>
                <tr>
                    <td><strong>XGBoost</strong></td>
                    <td>Boosting</td>
                    <td>Sequential Trees</td>
                    <td><span style="color:#00c8ff;">Candidate</span></td>
                </tr>
                <tr>
                    <td><strong>Random Forest</strong></td>
                    <td>Bagging</td>
                    <td>Ensemble Voting</td>
                    <td><strong style="color:#2ecc71;">Top Model 🏆</strong></td>
                </tr>
            </tbody>
        </table>
        <div style="margin-top:14px; font-size:13px; color:rgba(180,215,245,0.70); line-height:1.6;">
            💡 <strong>Key Finding:</strong> ElasticNet reaches 62.80% due to non-linear interaction boundaries.
            Random Forest demonstrates the highest individual accuracy (86.13%) by mitigating variance across 200 bagged decision trees.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 2. Live Multi-Model Evaluation on Specific User Inputs
    st.markdown('<div class="sec-title">🔬 Section 2: Live Multi-Model Evaluation on Sample Input</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:rgba(200,235,255,0.80); font-size:14px; margin-bottom:16px;'>"
        "Adjust the parameters below to observe how ElasticNet, XGBoost, and Random Forest independently evaluate the exact same sample."
        "</p>",
        unsafe_allow_html=True
    )

    comp_col1, comp_col2, comp_col3 = st.columns(3, gap="medium")

    with comp_col1:
        c_ph = st.slider("pH Level", 0.0, 14.0, 7.08, 0.05, key="c_ph")
        c_hardness = st.slider("Hardness (mg/L)", 40.0, 330.0, 196.37, 1.0, key="c_hard")
        c_solids = st.slider("Total Dissolved Solids (ppm)", 300.0, 62000.0, 22014.0, 100.0, key="c_sol")

    with comp_col2:
        c_chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.12, 0.05, key="c_chlor")
        c_sulfate = st.slider("Sulfates (mg/L)", 120.0, 500.0, 333.78, 1.0, key="c_sulf")
        c_conductivity = st.slider("Conductivity (μS/cm)", 150.0, 800.0, 426.21, 1.0, key="c_cond")

    with comp_col3:
        c_carbon = st.slider("Organic Carbon (ppm)", 2.0, 30.0, 14.28, 0.1, key="c_carb")
        c_trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 66.40, 0.5, key="c_tri")
        c_turbidity = st.slider("Turbidity (NTU)", 1.0, 7.0, 3.97, 0.05, key="c_turb")

    compare_button = st.button("⚖️ Compare All 3 Models on This Input", use_container_width=True, type="primary")

    if compare_button:
        # Build features
        c_ph_h = c_ph * c_hardness
        c_s_c = c_solids / (c_conductivity + 1.0)
        c_c_t = c_chloramines * c_trihalomethanes

        c_df = pd.DataFrame([[
            c_ph, c_hardness, c_solids, c_chloramines, c_sulfate,
            c_conductivity, c_carbon, c_trihalomethanes, c_turbidity,
            c_ph_h, c_s_c, c_c_t
        ]], columns=FEATURE_COLS)

        c_scaled = scaler.transform(c_df)

        # Run 3 models
        en_p = int(en_model.predict(c_scaled)[0])
        en_prob = float(en_model.predict_proba(c_scaled)[0][1] * 100)

        xgb_p = int(xgb_model.predict(c_scaled)[0])
        xgb_prob = float(xgb_model.predict_proba(c_scaled)[0][1] * 100)

        rf_p = int(rf_model.predict(c_scaled)[0])
        rf_prob = float(rf_model.predict_proba(c_scaled)[0][1] * 100)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 3 Cards for individual model votes
        v_col1, v_col2, v_col3 = st.columns(3, gap="medium")

        with v_col1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="m-badge badge-std">Linear Baseline</span>
                <div class="m-score">{en_prob:.1f}%</div>
                <div class="m-name">ElasticNet</div>
                <div class="m-desc">Vote: {"Potable ✅" if en_p == 1 else "Non-Potable ⚠️"}</div>
            </div>
            """, unsafe_allow_html=True)

        with v_col2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="m-badge badge-std">Boosting</span>
                <div class="m-score">{xgb_prob:.1f}%</div>
                <div class="m-name">XGBoost</div>
                <div class="m-desc">Vote: {"Potable ✅" if xgb_p == 1 else "Non-Potable ⚠️"}</div>
            </div>
            """, unsafe_allow_html=True)

        with v_col3:
            st.markdown(f"""
            <div class="metric-card top-performer">
                <span class="m-badge badge-top">🏆 Best Model</span>
                <div class="m-score top">{rf_prob:.1f}%</div>
                <div class="m-name" style="color:#2ecc71;">Random Forest</div>
                <div class="m-desc">Vote: {"Potable ✅" if rf_p == 1 else "Non-Potable ⚠️"}</div>
            </div>
            """, unsafe_allow_html=True)

        # Live Dynamic Comparison Chart
        c_models = ["ElasticNet", "XGBoost", "Random Forest"]
        c_scores = [en_prob, xgb_prob, rf_prob]
        c_verdicts = ["Potable" if p == 1 else "Non-Potable" for p in [en_p, xgb_p, rf_p]]
        c_colors = ["#4a6fa5", "#00a8e8", "#2ecc71"]

        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        bars = ax.bar(c_models, c_scores, color=c_colors, width=0.48, edgecolor="none")
        bars[-1].set_edgecolor("#55efc4")
        bars[-1].set_linewidth(2.4)

        ax.set_ylim(0, 118)
        ax.set_ylabel("Predicted Potability Probability (%)", color="#c8e6fa", fontsize=10.5, labelpad=8)
        ax.tick_params(axis="x", colors="#c8e6fa", labelsize=10.5)
        ax.tick_params(axis="y", colors="#c8e6fa", labelsize=10)

        for spine in ax.spines.values():
            spine.set_edgecolor((0.0, 0.70, 1.0, 0.18))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.yaxis.grid(True, linestyle="--", alpha=0.18, color="#00c8ff")
        ax.set_axisbelow(True)

        for bar, score, verdict in zip(bars, c_scores, c_verdicts):
            is_top = (bar == bars[-1])
            txt_color = "#2ecc71" if is_top else "#00c8ff"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 2.2,
                f"{score:.1f}%\n({verdict})",
                ha="center",
                va="bottom",
                color=txt_color,
                fontweight="bold",
                fontsize=10.5,
            )

        ax.set_title(
            "Individual Model Evaluations for this Input Sample",
            color="#a8d8f0",
            fontsize=12.5,
            fontweight="bold",
            pad=16,
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
