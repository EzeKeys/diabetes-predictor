
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# Define the feature names used during training
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Prediction App")
st.markdown("This AI model predicts the likelihood of a person having diabetes based on key health parameters.")

# Collect input from user
st.sidebar.header("Patient Input Parameters")
inputs = []
for feature in features:
    value = st.sidebar.number_input(f"{feature}:", min_value=0.0, step=0.1, format="%.2f")
    inputs.append(value)

# Make prediction
if st.button("Predict Diabetes Risk"):
    input_array = np.array([inputs])
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("🔴 The patient is likely to have **diabetes**.")
    else:
        st.success("🟢 The patient is **not likely** to have diabetes.")

    st.markdown("---")
    st.caption("Note: This model is for educational purposes and should not replace medical advice.")
