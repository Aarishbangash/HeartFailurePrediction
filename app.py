import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set wide layout and page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Custom CSS for styling result text prominently
st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: 700;
        color: #d6336c;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 20px;
        color: #6c757d;
        text-align: center;
        margin-top: 0;
        margin-bottom: 40px;
    }
    .section-header {
        font-size: 26px;
        color: #198754;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
        border-bottom: 3px solid #198754;
        padding-bottom: 5px;
    }
    .prediction-success {
        color: #198754;  /* bootstrap success green */
        font-weight: 900;
        font-size: 48px;
        text-align: center;
        background-color: #d1e7dd;
        border-radius: 12px;
        padding: 20px;
        margin: 0 auto 30px auto;
        max-width: 500px;
        box-shadow: 0 0 15px #19875466;
    }
    .prediction-fail {
        color: #dc3545;  /* bootstrap danger red */
        font-weight: 900;
        font-size: 48px;
        text-align: center;
        background-color: #f8d7da;
        border-radius: 12px;
        padding: 20px;
        margin: 0 auto 30px auto;
        max-width: 500px;
        box-shadow: 0 0 15px #dc354566;
    }
    .metric-label {
        font-weight: 600;
        color: #0d6efd;
    }
    .footer {
        margin-top: 50px;
        font-size: 14px;
        text-align: center;
        color: #adb5bd;
    }
    </style>
    <h1 class="title">❤️ Heart Disease Prediction</h1>
    <p class="subtitle">Predict the likelihood of heart disease using Random Forest Classifier</p>
    """, unsafe_allow_html=True)

# Load saved model and preprocessing objects
@st.cache_resource
def load_artifacts():
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ordinal_encoder.pkl", "rb") as f:
        ord_encoder = pickle.load(f)
    with open("onehot_encoder.pkl", "rb") as f:
        onehot_encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, ord_encoder, onehot_encoder, scaler, feature_columns

model, ord_encoder, onehot_encoder, scaler, feature_cols = load_artifacts()

# Sidebar form for user input
st.sidebar.header("Patient Info - Enter to Predict")
def user_input_features():
    Age = st.sidebar.slider("Age (years)", 29, 77, 54, help="Age of the patient")
    Gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=0, help="Biological sex of the patient")
    ChestPainType = st.sidebar.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=0, help="Type of chest pain experienced")
    RestingBP = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Resting blood pressure")
    Cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 240, help="Cholesterol level")
    FastingBS = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["No", "Yes"], index=0, help="Fasting blood sugar level")
    RestingECG = st.sidebar.selectbox("Resting ECG Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], index=0, help="Resting electrocardiogram results")
    MaxHR = st.sidebar.slider("Max Heart Rate Achieved", 60, 202, 150, help="Max heart rate achieved during exercise")
    ExerciseAngina = st.sidebar.selectbox("Exercise Induced Angina", options=["No", "Yes"], index=0, help="Whether exercise induced angina occurred")
    Oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, 0.1, help="ST depression induced by exercise relative to rest")
    ST_Slope = st.sidebar.selectbox("ST Slope", options=["Up", "Flat", "Down"], index=0, help="Slope of the peak exercise ST segment")
    return {
        'Age': Age,
        'Gender': Gender,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }

input_data = user_input_features()
input_df = pd.DataFrame([input_data])

def preprocess_input(df_in):
    df = df_in.copy()
    numerical = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    for col in numerical:
        if col in df.columns:
            if df[col].min() <= 0:
                shift_amount = 1 - df[col].min()
                df[col] = df[col] + shift_amount

    st_slope_vals = np.array(df[['ST_Slope']])
    st_slope_enc = ord_encoder.transform(st_slope_vals)
    df['ST_Slope'] = st_slope_enc.flatten()

    num_values = df[numerical].values.astype(float)
    num_scaled = scaler.transform(num_values)
    df[numerical] = num_scaled

    cat_cols = ['Gender', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina']
    cat_values = df[cat_cols]
    cat_encoded = onehot_encoder.transform(cat_values)

    cat_encoded_df = pd.DataFrame(cat_encoded, columns=onehot_encoder.get_feature_names_out(cat_cols))

    processed_df = pd.concat([df[numerical], cat_encoded_df, df[['ST_Slope']]], axis=1)

    processed_df = processed_df.reindex(columns=feature_cols, fill_value=0)

    return processed_df

X_input = preprocess_input(input_df)
prediction = model.predict(X_input)[0]
# We get probabilities if needed, but not used in UI now
# prediction_proba = model.predict_proba(X_input)[0, 1]

# Display prediction with enhanced styling
if prediction == 1:
    st.markdown(f'<div class="prediction-fail">Disease Detected ❤️‍🩹</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="prediction-success">No Disease Detected 💓</div>', unsafe_allow_html=True)

# Using your real evaluation metrics from your model (replace here with your actual values)
accuracy = 0.8641304347826086
precision = 0.89
recall = 0.88
f1 = 0.8826291079812206

st.markdown("<h2 class='section-header'>Model Performance Metrics (Random Forest)</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# Remove classification report as per request

# Feature importance display
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.markdown("<h2 class='section-header'>Top 10 Important Features</h2>", unsafe_allow_html=True)

top_features = feature_importance_df.head(10)

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, palette='viridis')
ax.set_title("Feature Importance from Random Forest")
ax.grid(axis="x", linestyle="--", alpha=0.6)
st.pyplot(fig)

# Footer
st.markdown('<div class="footer">Developed with ❤️ by YourName &nbsp;&nbsp;|&nbsp;&nbsp; Dataset & Model Adapted</div>', unsafe_allow_html=True)
