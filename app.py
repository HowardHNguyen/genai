import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
import os
import urllib.request

# Function to download a file if it doesn’t exist
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# URLs for model files on GitHub
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/stacking_genai_model.pkl'
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/frmgham2.csv'

# Local paths for models
stacking_model_path = 'stacking_genai_model.pkl'

# Download model if it doesn’t exist
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

# Load the stacking model with validation
@st.cache_resource
def load_stacking_model():
    try:
        # Load the model
        model = joblib.load(stacking_model_path)
        # Check if it’s a StackingClassifier
        if not isinstance(model, StackingClassifier):
            st.error(f"Loaded model is of type {type(model)}, expected StackingClassifier.")
            return None
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()

# Debug info
if stacking_model:
    st.write(f"Loaded model type: {type(stacking_model)}")
    st.write("StackingClassifier loaded successfully.")
else:
    st.write("Model loading failed. Check the file or URL.")

# Define feature columns exactly as used during training
feature_columns = [
    'SEX', 'AGE', 'educ', 'CURSMOKE', 'CIGPDAY', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE',
    'GLUCOSE', 'HDLC', 'LDLC', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
]

# Title of the Page
st.title("CVD Prediction App by Howard Nguyen")
st.write("Enter your parameters and click Predict to get the results.")

# Sidebar for user input
st.sidebar.header("Enter Your Parameters")
sex = st.sidebar.selectbox("SEX (0 = Female, 1 = Male)", [0, 1], index=0)
age = st.sidebar.slider("AGE", 32.0, 81.0, 54.79)
educ = st.sidebar.slider("Education Level (educ)", 1.0, 4.0, 1.99)
cursmoke = st.sidebar.selectbox("Current Smoker (0 = No, 1 = Yes)", [0, 1], index=1)
cigpday = st.sidebar.slider("Cigarettes per Day", 0.0, 90.0, 0.0)
totchol = st.sidebar.slider("Total Cholesterol", 107.0, 696.0, 241.16)
sysbp = st.sidebar.slider("Systolic BP", 83.5, 295.0, 136.32)
diabp = st.sidebar.slider("Diastolic BP", 30.0, 159.0, 80.0)
bmi = st.sidebar.slider("BMI", 15.0, 59.0, 25.68)
heartrte = st.sidebar.slider("Heart Rate", 40.0, 120.0, 75.0)
glucose = st.sidebar.slider("Glucose", 50.0, 360.0, 50.0)
hdlc = st.sidebar.slider("HDL Cholesterol", 20.0, 100.0, 50.0)
ldlc = st.sidebar.slider("LDL Cholesterol", 20.0, 300.0, 50.0)
diabetes = st.sidebar.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1], index=0)
bpmeds = st.sidebar.selectbox("BP Meds (0 = No, 1 = Yes)", [0, 1], index=0)
prevchd = st.sidebar.selectbox("Prev CHD (0 = No, 1 = Yes)", [0, 1], index=0)
prevap = st.sidebar.selectbox("PREVAP (0 = No, 1 = Yes)", [0, 1], index=0)
prevmi = st.sidebar.selectbox("PREVMI (0 = No, 1 = Yes)", [0, 1], index=0)
prevstrk = st.sidebar.selectbox("PREVSTRK (0 = No, 1 = Yes)", [0, 1], index=0)
prevhyp = st.sidebar.selectbox("PREVHYP (0 = No, 1 = Yes)", [0, 1], index=0)

# Prepare input data
user_data = {
    'SEX': sex, 'AGE': age, 'educ': educ, 'CURSMOKE': cursmoke, 'CIGPDAY': cigpday,
    'TOTCHOL': totchol, 'SYSBP': sysbp, 'DIABP': diabp, 'BMI': bmi, 'HEARTRTE': heartrte,
    'GLUCOSE': glucose, 'HDLC': hdlc, 'LDLC': ldlc, 'DIABETES': diabetes, 'BPMEDS': bpmeds,
    'PREVCHD': prevchd, 'PREVAP': prevap, 'PREVMI': prevmi, 'PREVSTRK': prevstrk,
    'PREVHYP': prevhyp
}
input_df = pd.DataFrame([user_data], columns=feature_columns)

# Processing Button
if st.button("Predict"):
    if stacking_model is None:
        st.error("Cannot make predictions: Model failed to load.")
    else:
        try:
            # Prediction using stacking model
            stacking_proba = stacking_model.predict_proba(input_df)[:, 1]
            st.write(f"**Stacking Model Prediction: CVD Risk Probability = {stacking_proba[0]:.2f}**")

            # Prediction Probability Distribution
            st.subheader("Prediction Probability Distribution")
            fig, ax = plt.subplots()
            bar = ax.barh(["Stacking Model"], [stacking_proba[0]], color="blue")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            # Add percentage label to the bar
            for rect in bar:
                width = rect.get_width()
                ax.text(width + 0.01, rect.get_y() + rect.get_height()/2, f"{width*100:.0f}%", va="center")
            st.pyplot(fig)

            # Model Performance
            st.subheader("Model Performance")
            st.write("The model has been evaluated on a test dataset with an AUC of 0.96.")

            # Notes
            st.subheader("Notes")
            st.write("""
                - These predictions are for informational purposes only.
                - Consult a healthcare professional for medical advice.
                - The model uses a stacking approach with multiple features.
            """, unsafe_allow_html=True)

        except AttributeError as e:
            st.error(f"Model error: {e}. The loaded model may not support 'predict_proba'. Check the model file.")
        except Exception as e:
            st.error(f"Error processing predictions or plotting: {e}")
