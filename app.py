import streamlit as st
import pandas as pd
import pickle

# Load the model and transformers
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("ordinal_encoder.pkl", "rb") as f:
    ord_encoder = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Streamlit UI
st.title("Heart Disease Prediction App")

st.write("Enter the patient information below:")

# Collect user inputs
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
restingbp = st.number_input("Resting Blood Pressure", min_value=0, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, value=200)
fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.number_input("Max Heart Rate", min_value=0, value=150)
angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, step=0.1, value=1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction logic
if st.button("Predict"):

    # Step 1: Create input DataFrame
    input_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingBP": restingbp,
        "Cholesterol": cholesterol,
        "FastingBS": fastingbs,
        "RestingECG": restecg,
        "MaxHR": maxhr,
        "ExerciseAngina": angina,
        "Oldpeak": oldpeak,
        "ST_Slope": slope
    }

    input_df = pd.DataFrame([input_data])

    # Step 2: Separate features
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina']
    ord_col = ['ST_Slope']

    # Step 3: Preprocess
    try:
        input_num = pd.DataFrame(scaler.transform(input_df[num_cols]), columns=num_cols)
        input_cat = pd.DataFrame(ohe.transform(input_df[cat_cols]).toarray(), columns=ohe.get_feature_names_out(cat_cols))
        input_slope = pd.DataFrame(ord_encoder.transform(input_df[ord_col]), columns=ord_col)

        # Step 4: Combine & reorder
        final_input = pd.concat([input_num, input_cat, input_slope], axis=1)
        final_input = final_input.reindex(columns=feature_columns, fill_value=0)

        # Step 5: Predict
        prediction = model.predict(final_input)[0]

        if prediction == 1:
            st.error("⚠️ The model predicts a risk of heart disease.")
        else:
            st.success("✅ The model predicts no heart disease risk.")
    
    except Exception as e:
        st.error("An error occurred during prediction. Check input and model files.")
        st.text(f"Error: {e}")
