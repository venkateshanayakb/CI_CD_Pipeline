import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Enter property details to get predicted price")

# Inputs
area = st.number_input("Area (sq ft)", min_value=300.0, max_value=5000, value=1200, step=50.0)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)

# Predict button
if st.button("Predict Price"):

    payload = {
        "area": area,
        "bedrooms": int(bedrooms)
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            price = result["Predicted_price"]

            st.success(f"Predicted Price: ₹ {price:,.2f}")

        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Could not connect to API: {e}")
