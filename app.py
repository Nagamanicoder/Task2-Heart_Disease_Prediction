import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load trained model
model = joblib.load("best_heart_model.pkl")

# Page setup
st.set_page_config(page_title="❤️ Heart Disease Prediction", page_icon="💓", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 1em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #ff4b2b, #ff416c);
        transform: scale(1.05);
    }
    .big-font {
        font-size:22px !important;
        font-weight:600;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Fill patient details here ⬇️")

# Main header
st.title("💓 Heart Disease Prediction Dashboard")
st.markdown("Enter patient details to check the **risk of heart disease**.")

# Input form in 2 columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    restecg = st.selectbox("Resting ECG", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 70, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

# Predict button
if st.button("🔍 Predict Risk"):
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                         exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease**")
        st.metric(label="Confidence", value=f"{proba:.2f}")
    else:
        st.success(f"✅ **No Heart Disease Detected**")
        st.metric(label="Confidence", value=f"{1-proba:.2f}")

    # Gauge-like chart for probability
    st.subheader("📊 Risk Probability")
    fig, ax = plt.subplots(figsize=(6,2))
    ax.barh(["Risk"], [proba], color="#ff4b2b")
    ax.barh(["Risk"], [1-proba], left=[proba], color="#4CAF50")
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    st.pyplot(fig)
