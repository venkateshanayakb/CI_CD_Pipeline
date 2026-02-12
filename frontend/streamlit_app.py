import streamlit as st
import requests
import os

# Backend endpoint
API_URL = os.getenv(
    "BACKEND_URL",
    "https://ci-cd-pipeline-kp92.onrender.com/predict"
)

st.set_page_config(page_title="Medicare Payment Predictor", layout="centered")

st.title("🏥 Medicare Reimbursement Prediction")
st.write("Predict average Medicare payment based on DRG, state, and discharge volume.")

# -------------------------------
# Inputs
# -------------------------------

total_discharges = st.number_input(
    "Total Discharges",
    min_value=1,
    max_value=5000,
    value=50,
    step=1
)

provider_state = st.text_input(
    "Provider State (e.g., NY, CA, TX)",
    value="NY"
)

drg_definition = st.text_input(
    "DRG Definition",
    value="194 - SIMPLE PNEUMONIA & PLEURISY W CC"
)

# -------------------------------
# Predict Button
# -------------------------------

if st.button("Predict Medicare Payment"):

    payload = {
        "total_discharges": int(total_discharges),
        "provider_state": provider_state.strip().upper(),
        "drg_definition": drg_definition.strip()
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            payment = result["predicted_medicare_payment"]

            st.success(f"Predicted Medicare Payment: $ {payment:,.2f}")

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
