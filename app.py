import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model or preprocessors: {e}")
    st.stop()

st.title("❤️ Heart Disease Risk Predictor")
st.write("Fill in the patient details below to predict the likelihood of heart disease.")

# Input fields
Age = st.number_input("Age", min_value=1, max_value=120, value=40)
RestingBP = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
MaxHR = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Categorical inputs - MATCHED with training column names
Gender = st.selectbox("Gender", ["Male", "Female"])
ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
ExerciseAngina = st.selectbox("Exercise-induced Angina", ["Y", "N"])
Slope = st.selectbox("Slope of ST segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    try:
        # Separate input into numerical and categorical
        num_data = pd.DataFrame([[Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak]],
                                columns=["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"])
        
        cat_data = pd.DataFrame([[Gender, ChestPainType, ExerciseAngina]],
                                columns=["Gender", "ChestPainType", "ExerciseAngina"])

        slope_data = pd.DataFrame([[Slope]], columns=["Slope"])  # separate to match encoder

        # Scale numeric
        scaled_num = pd.DataFrame(scaler.transform(num_data), columns=num_data.columns)

        # Encode categorical
        encoded_cat = pd.DataFrame(encoder.transform(cat_data).toarray(), columns=encoder.get_feature_names_out(cat_data.columns))
        encoded_slope = pd.DataFrame(encoder.transform(slope_data).toarray(), columns=encoder.get_feature_names_out(slope_data.columns))

        # Combine all
        final_input = pd.concat([scaled_num, encoded_cat, encoded_slope], axis=1)

        # Align with trained feature columns
        for col in feature_columns:
            if col not in final_input.columns:
                final_input[col] = 0
        final_input = final_input[feature_columns]

        # Predict
        prediction = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High risk of heart disease. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low risk of heart disease. (Probability: {prob:.2f})")

    except Exception as e:
        st.error("An error occurred during prediction. Check input and model files.")
        st.text(f"Error: {e}")
