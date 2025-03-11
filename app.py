import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.models import load_model
from sklearn.ensemble import StackingClassifier
import os
import urllib.request

# Define the CNN model function (only needed when loading models)
def create_cnn_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
    
    model = Sequential([
        Conv1D(16, kernel_size=3, activation='relu', input_shape=(19, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to download a file if it doesn't exist
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")

# URLs for model files on GitHub
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/genai_stacking_model.pkl'
cnn_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/cnn_model.h5'
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/frmgham2.csv'

# Local paths for models
stacking_model_path = 'genai_stacking_model.pkl'
cnn_model_path = 'cnn_model.h5'

# Download models if they don't exist
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

if not os.path.exists(cnn_model_path):
    st.info(f"Downloading {cnn_model_path}...")
    download_file(cnn_model_url, cnn_model_path)

# Load the stacking model
@st.cache_resource
def load_stacking_model():
    try:
        model = joblib.load("genai_stacking_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()

# Define feature columns exactly as used during training
feature_columns = ['SEX', 'AGE', 'educ', 'CURSMOKE', 'CIGPDAY', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 
                   'HEARTRTE', 'GLUCOSE', 'HDLC', 'LDLC', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 
                   'PREVMI', 'PREVSTRK', 'PREVHYP']

# Sidebar for user input
st.sidebar.header("Enter Your Parameters")
sex = st.sidebar.selectbox("SEX (0 = Female, 1 = Male)", [0, 1], index=0)
age = st.sidebar.slider("AGE", 32.0, 81.0, 54.79)
educ = st.sidebar.slider("EDUC (Education Level)", 1.0, 4.0, 1.99)
cursmoke = st.sidebar.selectbox("CURSMOKE (0 = No, 1 = Yes)", [0, 1], index=1)
cigpday = st.sidebar.slider("CIGPDAY (Cigarettes per Day)", 0.0, 90.0, 8.25)
totchol = st.sidebar.slider("TOTCHOL (Total Cholesterol)", 107.0, 696.0, 241.16)
sysbp = st.sidebar.slider("SYSBP (Systolic Blood Pressure)", 83.5, 295.0, 136.32)
diabp = st.sidebar.slider("DIABP (Diastolic Blood Pressure)", 30.0, 150.0, 80.0)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0)
heartrte = st.sidebar.slider("HEARTRTE (Heart Rate)", 40.0, 120.0, 75.0)
glucose = st.sidebar.slider("GLUCOSE", 50.0, 300.0, 100.0)
hdlc = st.sidebar.slider("HDLC (HDL Cholesterol)", 20.0, 100.0, 50.0)
ldlc = st.sidebar.slider("LDLC (LDL Cholesterol)", 50.0, 300.0, 130.0)
diabetes = st.sidebar.selectbox("DIABETES (0 = No, 1 = Yes)", [0, 1], index=0)
bpmeds = st.sidebar.selectbox("BPMEDS (0 = No, 1 = Yes)", [0, 1], index=0)
prevchd = st.sidebar.selectbox("PREVCHD (0 = No, 1 = Yes)", [0, 1], index=0)
prevap = st.sidebar.selectbox("PREVAP (0 = No, 1 = Yes)", [0, 1], index=0)
prevmi = st.sidebar.selectbox("PREVMI (0 = No, 1 = Yes)", [0, 1], index=0)
prevstrk = st.sidebar.selectbox("PREVSTRK (0 = No, 1 = Yes)", [0, 1], index=0)
prevhyp = st.sidebar.selectbox("PREVHYP (0 = No, 1 = Yes)", [0, 1], index=0)
hyperten = st.sidebar.selectbox("HYPERTEN (0 = No, 1 = Yes)", [0, 1], index=0)
stroke = st.sidebar.selectbox("STROKE (0 = No, 1 = Yes)", [0, 1], index=0)

# Prepare input data
user_data = {
    'SEX': sex, 'AGE': age, 'educ': educ, 'CURSMOKE': cursmoke, 'CIGPDAY': cigpday,
    'TOTCHOL': totchol, 'SYSBP': sysbp, 'DIABP': diabp, 'BMI': bmi, 'HEARTRTE': heartrte,
    'GLUCOSE': glucose, 'HDLC': hdlc, 'LDLC': ldlc, 'DIABETES': diabetes, 'BPMEDS': bpmeds,
    'PREVCHD': prevchd, 'PREVAP': prevap, 'PREVMI': prevmi, 'PREVSTRK': prevstrk,
    'PREVHYP': prevhyp, 'HYPERTEN': hyperten, 'STROKE': stroke
}
input_df = pd.DataFrame([user_data], columns=feature_columns)

# Predictions
# Make predictions
if stacking_model:
    try:
        stacking_proba = stacking_model.predict_proba(input_df)[:, 1]
        st.subheader("Predictions")
        st.write(f"Stacking Model Prediction: CVD Risk Probability = {stacking_proba[0]:.2f}")
        
        fig, ax = plt.subplots()
        ax.bar(["CVD Risk"], [stacking_proba[0]], color="blue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error making predictions: {e}")

    # Feature importances (optional)
    st.subheader("Feature Importances (XGBoost)")
    try:
        xgb_model = stacking_model.named_estimators_['xgb']  # Adjust 'xgb' if needed
        importances = xgb_model.feature_importances_
        fig, ax = plt.subplots()
        indices = np.argsort(importances)
        ax.barh(range(len(indices)), importances[indices], color='blue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting feature importances: {e}")

    # Model performance
    st.subheader("Model Performance")
    st.write("ROC curve not available for single prediction. Probability shown above.")
else:
    st.error("Model not loaded successfully.")