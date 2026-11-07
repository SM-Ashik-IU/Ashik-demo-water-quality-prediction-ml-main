import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pickle
import numpy as np
import streamlit as st
from config import MODEL_PATH

st.set_page_config(page_title="PureSip — Prediction", page_icon="🔬", layout="wide")

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

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #132235 100%);
    border-right: 1px solid rgba(100,180,255,0.14);
}
[data-testid="stSidebar"] * { color: #c8e6fa !important; }
[data-testid="stSidebar"] .stSlider > div [data-testid="stTickBar"] { opacity: 0; }
[data-testid="stSidebar"] .stSlider .stMarkdown { color: rgba(180,215,245,0.65) !important; }
/* Slider track accent */
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #0078d4 !important;
}

/* ── Page header ── */
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
.page-header-icon  { font-size: 42px; }
.page-header-title { font-size: 32px; font-weight: 800; color: #a8d8f0; letter-spacing: -0.5px; }
.page-header-sub   { font-size: 14px; color: rgba(180,215,245,0.60); margin-top: 4px; }

/* ── Divider ── */
.puresip-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,180,255,0.4), transparent);
    margin: 20px 0;
}

/* ── Section title ── */
.section-title {
    font-size: 17px; font-weight: 700; color: #a8d8f0;
    padding: 8px 0 6px;
    border-bottom: 1px solid rgba(0,180,255,0.18);
    margin-bottom: 16px;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(0,30,70,0.30);
    border: 1px solid rgba(0,180,255,0.16) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #a8d8f0 !important;
    font-weight: 600;
    font-size: 14px;
}
[data-testid="stExpander"] p { color: rgba(200,235,255,0.75) !important; }

/* ── Parameter value table ── */
.param-table { width: 100%; border-collapse: collapse; }
.param-table tr { border-bottom: 1px solid rgba(0,180,255,0.08); }
.param-table tr:last-child { border-bottom: none; }
.param-table td {
    padding: 9px 6px;
    font-size: 13.5px;
    color: rgba(200,235,255,0.80);
}
.param-table td:first-child {
    font-weight: 600;
    color: #a8d8f0;
    width: 55%;
}
.param-table td:last-child {
    font-family: 'Courier New', monospace;
    color: #00c8ff;
    text-align: right;
    font-size: 13px;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0062b8, #0078d4, #00a0e8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(0,120,212,0.35) !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #0078d4, #00a0e8, #00c8ff) !important;
    box-shadow: 0 6px 28px rgba(0,180,255,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Result cards ── */
.result-safe {
    background: linear-gradient(135deg, rgba(0,160,80,0.18), rgba(0,120,60,0.12));
    border: 2px solid rgba(0,200,100,0.45);
    border-radius: 16px;
    padding: 32px 28px;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
.result-unsafe {
    background: linear-gradient(135deg, rgba(200,40,40,0.18), rgba(160,20,20,0.12));
    border: 2px solid rgba(255,80,80,0.45);
    border-radius: 16px;
    padding: 32px 28px;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }

.result-verdict {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
    line-height: 1.2;
}
.result-verdict-safe   { color: #2ecc71; }
.result-verdict-unsafe { color: #ff6b6b; }
.result-icon   { font-size: 56px; margin-bottom: 14px; display: block; }
.result-sub    { font-size: 13.5px; color: rgba(200,235,255,0.65); margin-top: 10px; line-height: 1.6; }

/* ── Confidence strip ── */
.confidence-label {
    font-size: 13px;
    color: rgba(180,215,245,0.65);
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-top: 20px;
    margin-bottom: 6px;
}
.confidence-value {
    font-size: 36px;
    font-weight: 800;
    color: #00c8ff;
    line-height: 1;
}
/* stProgress bar accent */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0062b8, #00c8ff) !important;
    border-radius: 100px !important;
}
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 100px !important;
    height: 10px !important;
}

/* ── Info box ── */
.info-box {
    background: rgba(0,80,140,0.20);
    border-left: 4px solid #00c8ff;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    font-size: 13.5px;
    color: rgba(200,235,255,0.80);
    line-height: 1.6;
}

/* ── Sidebar header label ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a8d8f0 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-header-icon">🔬</div>
    <div>
        <div class="page-header-title">Water Quality Prediction</div>
        <div class="page-header-sub">PureSip &nbsp;·&nbsp; Stacking Ensemble (RF + XGBoost + GBM)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model  ◀ ORIGINAL LOGIC ─────────────────────────────────────────────
MODEL_DIR   = os.path.dirname(MODEL_PATH)
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

# ── Sidebar sliders  ◀ ORIGINAL LOGIC ────────────────────────────────────────
st.sidebar.header("Input Parameters")

def user_input():
    ph               = st.sidebar.slider("pH",                    0.0,   14.0,   7.0)
    hardness         = st.sidebar.slider("Hardness (mg/L)",       0.0,  500.0, 200.0)
    solids           = st.sidebar.slider("Solids (ppm)",          0.0,50000.0,20000.0)
    chloramines      = st.sidebar.slider("Chloramines (ppm)",     0.0,   15.0,   7.0)
    sulfate          = st.sidebar.slider("Sulfate (mg/L)",        0.0,  500.0, 300.0)
    conductivity     = st.sidebar.slider("Conductivity (μS/cm)",  0.0, 1000.0, 500.0)
    organic_carbon   = st.sidebar.slider("Organic Carbon (ppm)",  0.0,   30.0,  15.0)
    trihalomethanes  = st.sidebar.slider("Trihalomethanes (μg/L)",0.0,  150.0,  80.0)
    turbidity        = st.sidebar.slider("Turbidity (NTU)",       0.0,   10.0,   4.0)

    ph_hardness                 = ph * hardness
    solids_conductivity         = solids / (conductivity + 1)
    chloramines_trihalomethanes = chloramines * trihalomethanes

    return np.array([[ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes,
                      turbidity, ph_hardness, solids_conductivity,
                      chloramines_trihalomethanes]])

# ── Parameter glossary  ◀ ORIGINAL LOGIC (expander) ──────────────────────────
with st.expander("📖  What do these parameters mean?", expanded=False):
    st.markdown("""
    | Parameter | Description | WHO Safe Range |
    |---|---|---|
    | **pH** | Acidity/alkalinity of water | 6.5 – 8.5 |
    | **Hardness** | Calcium & magnesium content (mg/L) | < 300 mg/L |
    | **Solids** | Total dissolved solids (ppm) | < 500 ppm |
    | **Chloramines** | Disinfectant level (ppm) | < 4 ppm |
    | **Sulfate** | Naturally occurring mineral (mg/L) | < 250 mg/L |
    | **Conductivity** | Electrical conductivity (μS/cm) | < 400 μS/cm |
    | **Organic Carbon** | Carbon from organic matter (ppm) | < 2 ppm |
    | **Trihalomethanes** | Chlorination byproduct (μg/L) | < 80 μg/L |
    | **Turbidity** | Water cloudiness / clarity (NTU) | < 5 NTU |
    """)

st.markdown('<div class="puresip-divider"></div>', unsafe_allow_html=True)

# ── Collect input + scale  ◀ ORIGINAL LOGIC ──────────────────────────────────
input_data   = user_input()
input_scaled = scaler.transform(input_data)

# ── Two-column layout ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── Left — Input summary ──────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="section-title">📋 Current Input Values</div>', unsafe_allow_html=True)

    labels = ["pH", "Hardness", "Solids", "Chloramines", "Sulfate",
              "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"]

    rows = "".join(
        f"<tr><td>{label}</td><td>{round(val, 3)}</td></tr>"
        for label, val in zip(labels, input_data[0][:9])
    )
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(0,180,255,0.14);
                border-radius:12px; padding:16px 20px;">
        <table class="param-table">{rows}</table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:18px;">
        🎛️ &nbsp;Adjust the <strong>sliders in the sidebar</strong> to change
        the water parameters. The values above update in real time.
    </div>
    """, unsafe_allow_html=True)

# ── Right — Predict + Result  ◀ ORIGINAL LOGIC ───────────────────────────────
with col2:
    st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if st.button("🔍  Predict Water Quality", use_container_width=True):
        # ── ORIGINAL LOGIC — untouched ──
        prediction = model.predict(input_scaled)
        proba      = model.predict_proba(input_scaled)[0]
        confidence = round(max(proba) * 100, 2)

        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-safe">
                <span class="result-icon">✅</span>
                <div class="result-verdict result-verdict-safe">
                    Water is Safe for Drinking
                </div>
                <div class="result-sub">
                    The PureSip model classifies this water sample as
                    <strong>Potable</strong>. It meets the predicted
                    safety thresholds for the given parameters.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unsafe">
                <span class="result-icon">⚠️</span>
                <div class="result-verdict result-verdict-unsafe">
                    Water is NOT Safe for Drinking
                </div>
                <div class="result-sub">
                    The PureSip model classifies this water sample as
                    <strong>Not Potable</strong>. One or more parameters
                    may exceed safe thresholds. Do not consume without treatment.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence display ◀ ORIGINAL LOGIC ──
        st.markdown(f"""
        <div class="confidence-label">Model Confidence</div>
        <div class="confidence-value">{confidence}%</div>
        """, unsafe_allow_html=True)
        st.progress(confidence / 100)

    else:
        # Idle state placeholder
        st.markdown("""
        <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(0,180,255,0.12);
                    border-radius:14px; padding:52px 24px; text-align:center;">
            <div style="font-size:48px; margin-bottom:16px;">💧</div>
            <div style="font-size:16px; font-weight:600; color:#a8d8f0; margin-bottom:8px;">
                Ready to Analyze
            </div>
            <div style="font-size:13px; color:rgba(180,215,245,0.55);">
                Set water parameters using the sidebar sliders<br>
                then click <strong style="color:#00c8ff;">Predict</strong> above.
            </div>
        </div>
        """, unsafe_allow_html=True)
