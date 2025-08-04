import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model & preprocessors
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("onehot_encoder.pkl", "rb") as f:
        onehot_encoder = pickle.load(f)
    with open("ordinal_encoder.pkl", "rb") as f:
        ordinal_encoder = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model or preprocessors: {e}")
    st.stop()

st.title("❤️ Heart Disease Prediction App")

# Form input
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
        # Create DataFrame
        df = pd.DataFrame({
            "Age": [age],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "MaxHR": [max_hr],
            "Oldpeak": [oldpeak],
            "Gender": [gender],
            "ChestPainType": [cp_type],
            "FastingBS": [1 if fasting_bs == "Yes" else 0],
            "RestingECG": [rest_ecg],
            "ExerciseAngina": ["Y" if exercise_angina == "Yes" else "N"],
            "ST_Slope": [st_slope]
        })

        # Apply encoders
        df[["ST_Slope"]] = ordinal_encoder.transform(df[["ST_Slope"]])
        cat_features = ["Gender", "ChestPainType", "RestingECG", "ExerciseAngina"]
        onehot_df = pd.DataFrame(onehot_encoder.transform(df[cat_features]).toarray(),
                                 columns=onehot_encoder.get_feature_names_out(cat_features))

        # Combine numeric + encoded
        df_final = pd.concat([
            df[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS", "ST_Slope"]],
            onehot_df
        ], axis=1)

        # Ensure all expected columns exist
        for col in feature_columns:
            if col not in df_final.columns:
                df_final[col] = 0

        df_final = df_final[feature_columns]  # reorder

        # Scale
        df_scaled = scaler.transform(df_final)

        # Predict
        prediction = model.predict(df_scaled)[0]
        result = "Positive for Heart Disease 💔" if prediction == 1 else "No Heart Disease Detected ❤️"
        st.success(f"✅ Prediction: {result}")

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.error(f"Error: {str(e)}")
