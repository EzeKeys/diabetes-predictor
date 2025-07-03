import streamlit as st
import numpy as np
import joblib

# Load model and features
model = joblib.load("diabetes_model.pkl")
features = joblib.load("model_features.pkl")

# Page setup
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter the patient data below to predict diabetes risk.")

# Collect user input
inputs = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", min_value=0.0, step=0.1)
    inputs.append(value)

# Predict button
if st.button("🔍 Predict Diabetes Risk"):
    input_array = np.array([inputs])
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("🔴 The patient is likely to have **diabetes**.")
    else:
        st.success("🟢 The patient is unlikely to have **diabetes**.")

# Footer
st.markdown("---")
st.caption("Built by Ezekeys with ❤️ using Streamlit and XGBoost")
