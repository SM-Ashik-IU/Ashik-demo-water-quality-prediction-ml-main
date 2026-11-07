import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH

st.set_page_config(page_title="PureSip — EDA", page_icon="📊", layout="wide")

# ── Global CSS (shared theme) ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e8f4fd;
}
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #132235 100%);
    border-right: 1px solid rgba(100,180,255,0.12);
}
[data-testid="stSidebar"] * { color: #c8e6fa !important; }

/* Page header */
.page-header {
    background: linear-gradient(90deg, rgba(0,120,212,0.18), rgba(0,200,255,0.06));
    border: 1px solid rgba(0,180,255,0.18);
    border-radius: 16px;
    padding: 26px 32px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 18px;
    backdrop-filter: blur(8px);
}
.page-header-icon { font-size: 42px; }
.page-header-title {
    font-size: 32px; font-weight: 800; color: #a8d8f0;
    letter-spacing: -0.5px; line-height: 1.1;
}
.page-header-sub {
    font-size: 14px; color: rgba(180,215,245,0.60);
    margin-top: 4px; font-weight: 400;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(0,120,212,0.10);
    border: 1px solid rgba(0,180,255,0.18);
    border-radius: 12px;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"]  { color: rgba(180,215,245,0.65) !important; font-size: 13px !important; text-transform: uppercase; letter-spacing: 0.8px; }
[data-testid="stMetricValue"]  { color: #00c8ff !important; font-size: 32px !important; font-weight: 800 !important; }

/* Section headers */
.section-title {
    font-size: 18px; font-weight: 700; color: #a8d8f0;
    padding: 10px 0 6px;
    border-bottom: 1px solid rgba(0,180,255,0.18);
    margin-bottom: 18px;
}

/* Divider */
.puresip-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,180,255,0.4), transparent);
    margin: 24px 0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,180,255,0.15);
    border-radius: 10px;
    overflow: hidden;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: rgba(0,30,60,0.60) !important;
    border: 1px solid rgba(0,180,255,0.25) !important;
    color: #e8f4fd !important;
    border-radius: 8px !important;
}

/* Plot containers */
.plot-container {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(0,180,255,0.14);
    border-radius: 14px;
    padding: 20px;
    margin-top: 8px;
}

/* Info box */
.insight-box {
    background: rgba(0,80,140,0.20);
    border-left: 4px solid #00c8ff;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    font-size: 13.5px;
    color: rgba(200,235,255,0.80);
    line-height: 1.6;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-header-icon">📊</div>
    <div>
        <div class="page-header-title">Exploratory Data Analysis</div>
        <div class="page-header-sub">PureSip &nbsp;·&nbsp; Kaggle Water Potability Dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load data  ◀ ORIGINAL LOGIC ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# ── Overview metrics ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows",      df.shape[0])
col2.metric("Total Columns",   df.shape[1])
col3.metric("Missing Values",  df.isnull().sum().sum())

st.markdown('<div class="puresip-divider"></div>', unsafe_allow_html=True)

# ── Sample data ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗂️ Sample Data</div>', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)

st.markdown('<div class="puresip-divider"></div>', unsafe_allow_html=True)

# ── Class distribution  ◀ ORIGINAL LOGIC ─────────────────────────────────────
st.markdown('<div class="section-title">⚖️ Class Distribution</div>', unsafe_allow_html=True)
left, right = st.columns([1, 1])
with left:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    df["Potability"].value_counts().plot(kind="bar", ax=ax, color=["#3B8BD4", "#E85D24"])
    ax.set_xticklabels(["Not Potable", "Potable"], rotation=0, color="#c8e6fa", fontsize=11)
    ax.set_ylabel("Count", color="#c8e6fa")
    ax.tick_params(colors="#c8e6fa")
    for spine in ax.spines.values():
        spine.set_edgecolor((0.0, 0.70, 1.0, 0.20))
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    potable     = int(df["Potability"].sum())
    not_potable = len(df) - potable
    pct         = round(potable / len(df) * 100, 1)
    st.markdown(f"""
    <div style="padding:24px 0;">
        <div class="insight-box">
            📌 &nbsp;Out of <strong>{len(df):,}</strong> samples,
            <strong style="color:#3B8BD4;">{not_potable:,}</strong> are
            <span style="color:#3B8BD4;">Not Potable</span> and
            <strong style="color:#E85D24;">{potable:,}</strong> are
            <span style="color:#E85D24;">Potable</span> —
            representing <strong>{pct}%</strong> potability rate.<br><br>
            The dataset is <strong>imbalanced</strong>, which is why SMOTE
            oversampling is applied during training to balance the classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="puresip-divider"></div>', unsafe_allow_html=True)

# ── Feature distributions  ◀ ORIGINAL LOGIC ──────────────────────────────────
st.markdown('<div class="section-title">📈 Feature Distributions</div>', unsafe_allow_html=True)
feature = st.selectbox("Select Feature", df.columns[:-1])
st.markdown('<div class="plot-container">', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(9, 3.6))
fig.patch.set_facecolor("none")
ax.set_facecolor("none")
sns.histplot(df[feature], kde=True, ax=ax, color="#3B8BD4")
ax.tick_params(colors="#c8e6fa")
ax.set_xlabel(feature, color="#c8e6fa")
ax.set_ylabel("Count",  color="#c8e6fa")
for spine in ax.spines.values():
    spine.set_edgecolor((0.0, 0.70, 1.0, 0.20))
ax.lines[0].set_color("#00c8ff")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="puresip-divider"></div>', unsafe_allow_html=True)

# ── Correlation heatmap  ◀ ORIGINAL LOGIC ────────────────────────────────────
st.markdown('<div class="section-title">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
st.markdown('<div class="plot-container">', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("none")
ax.set_facecolor("none")
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            annot_kws={"size": 9, "color": "white"},
            linewidths=0.4, linecolor="#0078d4")
ax.tick_params(colors="#c8e6fa", labelsize=9)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-box" style="margin-top:16px;">
    💡 &nbsp;No single feature shows a strong linear correlation with
    <strong>Potability</strong>, confirming that an ensemble non-linear model
    is the right approach for this dataset.
</div>
""", unsafe_allow_html=True)
