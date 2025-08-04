import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved components
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("ordinal_encoder.pkl", "rb") as f:
    ord_encoder = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("💓 Heart Disease Prediction App")

# Input fields
st.header("Enter Patient Data")

age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["male", "female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.selectbox("Resting ECG", ["normal", "ST-T abnormality", "left ventricular hypertrophy"])
thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
exang = st.selectbox("Exercise Induced Angina", ["yes", "no"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

# Prepare input DataFrame
input_dict = {
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
}

input_df = pd.DataFrame(input_dict)

# Identify feature types
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ord_cols = ["slope", "ca", "thal"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang"]

# Preprocessing
input_num = pd.DataFrame(scaler.transform(input_df[num_cols]), columns=num_cols)
input_cat = pd.DataFrame(ohe.transform(input_df[cat_cols]).toarray(), columns=ohe.get_feature_names_out(cat_cols))
input_ord = pd.DataFrame(ord_encoder.transform(input_df[ord_cols]), columns=ord_cols)

# Combine processed parts
final_input = pd.concat([input_num, input_cat, input_ord], axis=1)

# Ensure same column order
final_input = final_input.reindex(columns=feature_columns, fill_value=0)

# Predict
if st.button("Predict"):
    pred = model.predict(final_input)[0]
    if pred == 1:
        st.error("❌ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")
