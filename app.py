import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load('heart_model.pkl')
ohe = joblib.load('onehot_encoder.pkl')
ord_encoder = joblib.load('ordinal_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# UI
st.title("❤️ Heart Disease Prediction App")

st.header("Enter Patient Information")
age = st.number_input("Age", min_value=1, max_value=120)
resting_bp = st.number_input("Resting Blood Pressure", min_value=0)
cholesterol = st.number_input("Cholesterol", min_value=0)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
max_hr = st.number_input("Maximum Heart Rate", min_value=0)
oldpeak = st.number_input("Oldpeak", min_value=0.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

gender = st.selectbox("Gender", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

# Create DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'MaxHR': [max_hr],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope],
    'Gender': [gender],
    'ChestPainType': [chest_pain],
    'RestingECG': [resting_ecg],
    'ExerciseAngina': [exercise_angina]
})

# Column groups
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
cat_cols = ['Gender', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'FastingBS']
ord_col = ['ST_Slope']

# Transform inputs
encoded_ord = ord_encoder.transform(input_df[ord_col])  # returns 2D array
encoded_cat = ohe.transform(input_df[cat_cols])         # returns sparse or array
scaled_num = scaler.transform(input_df[num_cols])       # returns array

# Combine all features
X = pd.concat([
    pd.DataFrame(scaled_num, columns=num_cols),
    pd.DataFrame(encoded_ord, columns=['ST_Slope']),
    pd.DataFrame(encoded_cat.toarray(), columns=ohe.get_feature_names_out(cat_cols))
], axis=1)

# Predict
if st.button("Predict"):
    pred = model.predict(X)[0]
    if pred == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")
