"""
Unit and Integration Tests for PureSip ML System
================================================
Tests model persistence, data transformation, feature engineering,
and single-model vs. comparative model inference.
"""

import os
import sys
import pickle
import ast
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PATH, MODEL_PATH


@pytest.fixture(scope="module")
def artifacts():
    """Load model, scaler, and dataset once for all tests."""
    scaler_path = os.path.join(os.path.dirname(MODEL_PATH), "scaler.pkl")
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    assert os.path.exists(scaler_path), f"Scaler file not found at {scaler_path}"
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"

    with open(MODEL_PATH, "rb") as f:
        stacking_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(DATA_PATH)
    return {
        "model": stacking_model,
        "scaler": scaler,
        "rf": stacking_model.named_estimators_["rf"],
        "xgb": stacking_model.named_estimators_["xgb"],
        "df": df,
    }


def test_app_syntax_and_structure():
    """Verify app.py parses without syntax errors and contains the 3 views."""
    app_path = os.path.join(os.path.dirname(__file__), "..", "app", "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()

    # AST syntax check
    ast.parse(code)

    # Architectural navigation checks
    assert 'NAV_OPTIONS = ["Home", "Prediction", "Comparison"]' in code or 'options=["Home", "Prediction", "Comparison"]' in code
    assert 'menu == "Home"' in code
    assert 'menu == "Prediction"' in code
    assert 'menu == "Comparison"' in code


def test_feature_engineering_consistency(artifacts):
    """Verify engineered interaction features match training dimensions (12 columns)."""
    df = artifacts["df"].copy()
    df.fillna(df.mean(), inplace=True)
    df["ph_hardness"] = df["ph"] * df["Hardness"]
    df["solids_conductivity"] = df["Solids"] / (df["Conductivity"] + 1)
    df["chloramines_trihalomethanes"] = df["Chloramines"] * df["Trihalomethanes"]

    X = df.drop("Potability", axis=1)
    expected_cols = [
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
        "ph_hardness", "solids_conductivity", "chloramines_trihalomethanes"
    ]
    assert list(X.columns) == expected_cols
    assert X.shape[1] == 12


def test_rf_single_model_inference(artifacts):
    """Verify single best model (Tuned Random Forest) outputs valid predictions and probabilities."""
    scaler = artifacts["scaler"]
    rf = artifacts["rf"]

    # Typical potable test sample
    sample = pd.DataFrame([[
        7.08, 196.37, 22014.0, 7.12, 333.78, 426.21, 14.28, 66.40, 3.97,
        7.08 * 196.37, 22014.0 / (426.21 + 1.0), 7.12 * 66.40
    ]], columns=[
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
        "ph_hardness", "solids_conductivity", "chloramines_trihalomethanes"
    ])

    scaled = scaler.transform(sample)
    pred = rf.predict(scaled)
    proba = rf.predict_proba(scaled)[0]

    assert pred[0] in [0, 1]
    assert len(proba) == 2
    assert np.isclose(np.sum(proba), 1.0)
    assert 0.0 <= proba[0] <= 1.0
    assert 0.0 <= proba[1] <= 1.0


def test_feature_importance_weights(artifacts):
    """Verify Random Forest provides non-zero feature importance weights."""
    rf = artifacts["rf"]
    importances = rf.feature_importances_
    assert len(importances) == 12
    assert np.isclose(np.sum(importances), 1.0)
    assert np.all(importances >= 0.0)
