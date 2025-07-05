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

# features 
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# sample inppit
example_input = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0.0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

# Form for user input
with st.form("wine_input_form"):
    cols = st.columns(3)
    user_input = {}

    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                value=example_input[feature],
                format="%.4f"
            )

    use_example = st.form_submit_button("Use Sample Input")
    predict_button = st.form_submit_button("Predict")

# Make prediction
if predict_button or use_example:
    input_array = np.array([user_input[f] for f in FEATURES]).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    confidence = round(100 * max(probabilities), 2)

    st.subheader("🔍 Prediction Result")
    st.success("✅ Good Quality Wine") if prediction == 1 else st.warning("❌ Not Good Quality Wine")
    st.write(f"**Confidence Score:** {confidence}%")
