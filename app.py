import streamlit as st
import numpy as np
import joblib
import os

# Load model
model_path = "models/xgb_model.joblib"
if not os.path.exists(model_path):
    st.error("Model not found. Please train the model and ensure it is saved in 'models/xgb_model.joblib'.")
    st.stop()

model = joblib.load(model_path)

st.title("🍷 Wine Quality Predictor")
st.write("Enter the chemical attributes of a wine sample below to predict if it's **Good** or **Not Good**.")

# Define features
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# Default values
good_sample = {
    'fixed acidity': 7.4, 'volatile acidity': 0.35, 'citric acid': 0.45, 'residual sugar': 2.1,
    'chlorides': 0.045, 'free sulfur dioxide': 35.0, 'total sulfur dioxide': 120.0,
    'density': 0.994, 'pH': 3.4, 'sulphates': 0.65, 'alcohol': 11.0
}

bad_sample = {
    'fixed acidity': 6.0, 'volatile acidity': 0.9, 'citric acid': 0.0, 'residual sugar': 1.0,
    'chlorides': 0.12, 'free sulfur dioxide': 5.0, 'total sulfur dioxide': 15.0,
    'density': 0.999, 'pH': 3.1, 'sulphates': 0.3, 'alcohol': 8.5
}

# Initialize session state
if 'inputs' not in st.session_state:
    st.session_state.inputs = {f: 0.0 for f in FEATURES}

# Buttons for loading sample data
col1, col2 = st.columns(2)
with col1:
    if st.button("🍇 Load Good Wine Sample"):
        st.session_state.inputs = good_sample.copy()
with col2:
    if st.button("🍷 Load Bad Wine Sample"):
        st.session_state.inputs = bad_sample.copy()

# Form input
with st.form("wine_input_form"):
    cols = st.columns(3)
    user_input = {}

    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                value=st.session_state.inputs.get(feature, 0.0),
                key=feature,
                format="%.4f"
            )

    predict_button = st.form_submit_button("Predict")

# Input array for prediction
input_array = np.array([user_input[f] for f in FEATURES]).reshape(1, -1)

# Prediction
if predict_button:
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    confidence = round(100 * max(probabilities), 2)

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.success("✅ Good Quality Wine")
    else:
        st.warning("❌ Not Good Quality Wine")
    st.write(f"**Confidence Score:** {confidence}%")
