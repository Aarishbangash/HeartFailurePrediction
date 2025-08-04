import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load saved models
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("💓 Heart Disease Prediction App")
st.markdown("This app uses a **Random Forest Classifier** to predict the likelihood of heart disease.")

with st.sidebar:
    st.header("User Input")
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0", "1"])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    ST_Slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

    # Numerical features
    Age = st.slider("Age", 20, 90, 50)
    RestingBP = st.slider("Resting BP", 80, 200, 120)
    Cholesterol = st.slider("Cholesterol", 100, 600, 200)
    MaxHR = st.slider("Maximum Heart Rate", 60, 220, 150)
    Oldpeak = st.slider("Oldpeak", 0.0, 7.0, 1.0, step=0.1)

if st.button("Predict"):

    # Prepare input
    input_df = pd.DataFrame([{
        'Gender': Gender,
        'ChestPainType': ChestPainType,
        'FastingBS': int(FastingBS),
        'RestingECG': RestingECG,
        'ExerciseAngina': ExerciseAngina,
        'ST_Slope': ST_Slope,
        'Age': Age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'MaxHR': MaxHR,
        'Oldpeak': Oldpeak
    }])

    # Encode categorical
    slope_encoded = ordinal_encoder.transform(input_df[['ST_Slope']])
    cat_encoded = onehot_encoder.transform(input_df[['Gender', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina']])
    num_scaled = scaler.transform(input_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])

    # Combine all
    final_input = np.concatenate([num_scaled, cat_encoded, slope_encoded], axis=1)

    # Prediction
    prediction = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🧠 Prediction")
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.metric("Prediction Probability", f"{prob*100:.2f}%")

    with col2:
        st.subheader("🔍 Feature Importance")
        importance = model.feature_importances_
        feature_names = list(scaler.feature_names_in_) + list(onehot_encoder.get_feature_names_out(['Gender', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina'])) + ['ST_Slope']

        sorted_idx = np.argsort(importance)[::-1]
        top_features = [feature_names[i] for i in sorted_idx][:10]
        top_importances = importance[sorted_idx][:10]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_features[::-1], top_importances[::-1], color='skyblue')
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)

    with st.expander("See Raw Input Data"):
        st.dataframe(input_df)

else:
    st.info("👈 Enter data from the sidebar and click **Predict**")
