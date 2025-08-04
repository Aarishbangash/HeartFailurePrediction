import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load('heart_model.pkl')
encoder = joblib.load('encoder.pkl')

st.title("Heart Disease Prediction App")

# User inputs
st.header("Enter Patient Details:")
age = st.number_input("Age", min_value=1, max_value=120, step=1)
resting_bp = st.number_input("Resting Blood Pressure", min_value=0)
cholesterol = st.number_input("Cholesterol", min_value=0)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
max_hr = st.number_input("Maximum Heart Rate", min_value=0)
oldpeak = st.number_input("Oldpeak", min_value=0.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])  # ordinal

gender = st.selectbox("Gender", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

# Encode ST Slope manually as ordinal
st_slope_map = {'Down': 0, 'Flat': 1, 'Up': 2}
st_slope_val = st_slope_map[st_slope]

# Create input DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'MaxHR': [max_hr],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope_val],
    'Gender': [gender],
    'ChestPainType': [chest_pain],
    'RestingECG': [resting_ecg],
    'ExerciseAngina': [exercise_angina],
    'FastingBS': [fasting_bs]
})

# Column types
cat_cols = ['Gender', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'FastingBS']
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'ST_Slope']

# Encode categorical columns
encoded_cat = encoder.transform(input_df[cat_cols])
encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

# Combine numerical and encoded categorical
final_input = pd.concat([input_df[num_cols].reset_index(drop=True), encoded_df], axis=1)

# Predict
if st.button("Predict"):
    prediction = model.predict(final_input)[0]
    if prediction == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")
