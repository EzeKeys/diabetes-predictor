import streamlit as st
import numpy as np
import joblib

# Educational content for features (excluding dropped features)
FEATURE_EDUCATION = {
    'pregnancies': {
        'title': 'Number of Pregnancies',
        'description': 'Total number of times the patient has been pregnant',
        'how_to_measure': 'Ask the patient directly about their pregnancy history',
        'normal_range': '0-17 pregnancies',
        'tips': 'Include all pregnancies (live births, stillbirths, miscarriages, abortions)'
    },
    'glucose': {
        'title': 'Plasma Glucose Concentration',
        'description': 'Blood glucose level after 2-hour oral glucose tolerance test',
        'how_to_measure': 'OGTT: Patient fasts overnight, drinks glucose solution, blood drawn after 2 hours',
        'normal_range': '<140 mg/dL (normal), 140-199 mg/dL (prediabetes), ≥200 mg/dL (diabetes)',
        'tips': 'Ensure patient fasts 8-12 hours before test. Avoid during illness or stress.'
    },
    'bloodpressure': {
        'title': 'Diastolic Blood Pressure',
        'description': 'Blood pressure measurement (mmHg) - bottom number in BP reading',
        'how_to_measure': 'Use calibrated sphygmomanometer. Patient seated, arm at heart level',
        'normal_range': '<80 mmHg (normal), 80-89 mmHg (stage 1), ≥90 mmHg (stage 2)',
        'tips': 'Take multiple readings, avoid caffeine/exercise 30 min before measurement'
    },
    'insulin': {
        'title': '2-Hour Serum Insulin',
        'description': 'Insulin level 2 hours after glucose load (mu U/ml)',
        'how_to_measure': 'Blood draw 2 hours after glucose tolerance test',
        'normal_range': '16-166 mu U/ml (normal response)',
        'tips': 'Coordinate with glucose tolerance test, ensure proper sample handling'
    },
    'bmi': {
        'title': 'Body Mass Index',
        'description': 'Weight in kg divided by height in meters squared',
        'how_to_measure': 'BMI = weight (kg) / height (m)²',
        'normal_range': '<25 (normal), 25-29.9 (overweight), ≥30 (obese)',
        'tips': 'Use calibrated scale, measure height without shoes, consistent timing'
    },
    'age': {
        'title': 'Age',
        'description': 'Patient age in years',
        'how_to_measure': 'Verify with identification document',
        'normal_range': 'Any age (risk increases with age)',
        'tips': 'Use actual age, not rounded. Consider screening frequency based on age'
    }
}

# Custom CSS for better styling
st.markdown("""
<style>
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .measurement-tip {
        background-color: #0000;
        color:#ffff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and features with error handling
@st.cache_resource
def load_model_data():
    try:
        model = joblib.load("diabetes_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure 'diabetes_model.pkl' and 'model_features.pkl' are in the same directory.")
        return None, None

# Page setup
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter the patient data below to predict diabetes risk.")

# Load model
model, features = load_model_data()

if model is None or features is None:
    st.stop()

# Show info about the tool
with st.expander("ℹ️ About this tool"):
    st.markdown("""
    <div class="info-box">
        <p>This tool uses machine learning to assess diabetes risk based on clinical measurements. 
        Each field below includes guidance on how to obtain accurate measurements.</p>
        <p><strong>Note:</strong> This is a screening tool only. Always confirm with appropriate laboratory tests and clinical evaluation.</p>
    </div>
    """, unsafe_allow_html=True)

# Collect user input with educational content
inputs = []
for feature in features:
    feature_lower = feature.lower()
    
    # Check if we have educational content for this feature
    if feature_lower in FEATURE_EDUCATION:
        edu_info = FEATURE_EDUCATION[feature_lower]
        
        st.markdown(f"### {edu_info['title']}")
        st.markdown(f"*{edu_info['description']}*")
        
        # Input field with appropriate constraints
        if feature_lower == 'age':
            value = st.number_input(
                f"Enter {feature}:",
                min_value=0,
                max_value=120,
                value=30,
                step=1,
                help=f"Normal range: {edu_info['normal_range']}"
            )
        elif feature_lower == 'pregnancies':
            value = st.number_input(
                f"Enter {feature}:",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help=f"Normal range: {edu_info['normal_range']}"
            )
        else:
            value = st.number_input(
                f"Enter {feature}:",
                min_value=0.0,
                value=0.0,
                step=0.1,
                help=f"Normal range: {edu_info['normal_range']}"
            )
        
        # Show measurement guidance
        st.markdown(f"""
        <div class="measurement-tip">
            <strong>📋 How to measure:</strong> {edu_info['how_to_measure']}<br>
            <strong>💡 Clinical tips:</strong> {edu_info['tips']}<br>
            <strong>📊 Normal range:</strong> {edu_info['normal_range']}
        </div>
        """, unsafe_allow_html=True)
        
        inputs.append(value)
        st.markdown("---")
    else:
        # Fallback for any features not in our education dictionary
        value = st.number_input(f"Enter {feature}:", min_value=0.0, step=0.1)
        inputs.append(value)

# Predict button
if st.button("🔍 Predict Diabetes Risk"):
    if all(x >= 0 for x in inputs):
        input_array = np.array([inputs])
        prediction = model.predict(input_array)[0]
        
        # Show prediction with additional context
        if prediction == 1:
            st.error("🔴 **High Risk: The patient is likely to have diabetes**")
            st.markdown("""
            **Recommended Next Steps:**
            - Perform confirmatory testing (HbA1c, fasting glucose)
            - Provide lifestyle counseling
            - Consider referral to endocrinologist
            - Schedule regular monitoring
            """)
        else:
            st.success("🟢 **Low Risk: The patient is unlikely to have diabetes**")
            st.markdown("""
            **Recommended Next Steps:**
            - Continue regular health screenings
            - Maintain healthy lifestyle habits
            - Annual diabetes risk assessment
            - Monitor for risk factor changes
            """)
        
        # Show probability if available
        try:
            prediction_proba = model.predict_proba(input_array)[0]
            risk_percentage = prediction_proba[1] * 100
            st.metric("Risk Probability", f"{risk_percentage:.1f}%")
        except:
            pass  # Some models may not have predict_proba
            
    else:
        st.error("❌ Please enter valid values for all fields.")

# Footer
st.markdown("---")
st.caption("Built by Ezekeys with ❤️ using Streamlit and XGBoost")
st.caption("⚠️ For educational and screening purposes only. Not a substitute for professional medical diagnosis.")
