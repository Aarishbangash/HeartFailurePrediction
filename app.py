import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Expected features
expected_cols = [
    'Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak',
    'Gender_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'FastingBS_Yes', 'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_Y', 'ST_Slope'
]

st.title("❤️ Heart Disease Prediction App")

# Input form
with st.form("user_inputs"):
    age = st.number_input("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure", 50, 200, 120)
    cholesterol = st.number_input("Cholesterol", 0, 600, 200)
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

    gender = st.selectbox("Gender", ["Male", "Female"])
    cp_type = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Base numeric features
        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }

        # One-hot encoded manual mappings
        if gender == "Male":
            input_data["Gender_M"] = 1
        else:
            input_data["Gender_M"] = 0

        for val in ["ATA", "NAP", "TA"]:
            input_data[f"ChestPainType_{val}"] = 1 if cp_type == val else 0

        input_data["FastingBS_Yes"] = 1 if fasting_bs == "Yes" else 0

        for val in ["Normal", "ST"]:
            input_data[f"RestingECG_{val}"] = 1 if rest_ecg == val else 0

        input_data["ExerciseAngina_Y"] = 1 if exercise_angina == "Yes" else 0

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all expected columns exist
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_cols]

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Positive for Heart Disease 💔" if prediction == 1 else "No Heart Disease Detected ❤️"
        st.success(f"✅ Prediction: {result}")

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.error(f"Error: {str(e)}")
