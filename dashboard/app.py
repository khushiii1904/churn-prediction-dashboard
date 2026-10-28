"""Streamlit dashboard that talks to the FastAPI prediction service."""
import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📉", layout="wide")
st.title("📉 Churn Prediction & Explainability Dashboard")
st.caption("XGBoost model · SHAP explanations · served via FastAPI")

# ---------------- sidebar: customer profile ----------------
with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 5)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 89.5, step=0.5)
    calls = st.slider("Support calls (last 6 mo)", 0, 10, 3)
    tech = st.checkbox("Tech support")
    sec = st.checkbox("Online security")
    paperless = st.checkbox("Paperless billing", value=True)
    senior = st.checkbox("Senior citizen")
    partner = st.checkbox("Has partner")
    deps = st.checkbox("Has dependents")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

yn = lambda b: "Yes" if b else "No"  # noqa: E731
payload = {
    "gender": gender, "senior_citizen": yn(senior), "partner": yn(partner), "dependents": yn(deps),
    "tenure": tenure, "contract": contract, "internet_service": internet,
    "payment_method": payment, "paperless_billing": yn(paperless),
    "tech_support": yn(tech), "online_security": yn(sec),
    "monthly_charges": monthly, "total_charges": round(monthly * max(tenure, 1), 2),
    "num_support_calls": calls,
}

tab_pred, tab_model = st.tabs(["🔮 Live prediction", "🧠 Model insights"])

with tab_pred:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
    except Exception as e:
        st.error(f"Could not reach the prediction API at {API_URL}\n\n{e}")
        st.stop()

    prob = res["churn_probability"]
    c1, c2 = st.columns([1, 2])
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": f"Churn risk: {res['risk_level']}"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#d62728" if prob >= 0.7 else "#ff7f0e" if prob >= 0.4 else "#2ca02c"},
                   "steps": [{"range": [0, 40], "color": "#e8f5e9"},
                             {"range": [40, 70], "color": "#fff3e0"},
                             {"range": [70, 100], "color": "#ffebee"}]}))
        gauge.update_layout(height=300, margin=dict(t=60, b=0, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)
        st.metric("Prediction", res["prediction"])
    with c2:
        st.subheader("Why this prediction? (SHAP)")
        factors = res["top_factors"][::-1]
        fig = go.Figure(go.Bar(
            x=[f["shap_value"] for f in factors], y=[f["feature"] for f in factors],
            orientation="h",
            marker_color=["#d62728" if f["shap_value"] > 0 else "#1f77b4" for f in factors]))
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          xaxis_title="SHAP value (log-odds impact) — red raises churn risk, blue lowers it")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw API response"):
        st.json(res)

with tab_model:
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=10).json()
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC", m["roc_auc"]); b.metric("PR-AUC", m["pr_auc"])
        c.metric("F1", m["f1"]); d.metric("Churn rate", f"{m['churn_rate']:.1%}")
        st.subheader("Global feature importance (mean |SHAP|)")
        imp = dict(list(m["global_shap_importance"].items())[:15])
        st.bar_chart(imp, horizontal=True)
    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")
