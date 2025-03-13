import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import urllib.request

# Function to download a file if it doesn’t exist
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")

# URLs for model files on GitHub
rf_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/rf_model.pkl'
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/frmgham2.csv'

# Local paths for models
rf_model_path = 'rf_model.pkl'

# Download models if they don’t exist
if not os.path.exists(rf_model_path):
    st.info(f"Downloading {rf_model_path}...")
    download_file(rf_model_url, rf_model_path)

# Load the Random Forest model
@st.cache_resource
def load_rf_model():
    try:
        model = joblib.load("rf_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

rf_model = load_rf_model()

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(data_url)
        data.fillna(data.mean(), inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

# Define feature columns exactly as used during training
feature_columns = [
    'SEX', 'AGE', 'educ', 'CURSMOKE', 'CIGPDAY', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE',
    'GLUCOSE', 'HDLC', 'LDLC', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
]

# Title of the Page
st.title("CVD Prediction App by Howard Nguyen")
st.write("Enter your parameters and click Predict to get the results.")

# Sidebar input parameters
st.sidebar.header('Enter Your Parameters')

def user_input_features():
    user_data = {}
    for feature in feature_columns:
        if feature in ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP']:
            user_data[feature] = st.sidebar.selectbox(feature, [0, 1])
        else:
            user_data[feature] = st.sidebar.slider(feature, float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
    
    return pd.DataFrame(user_data, index=[0])

input_df = user_input_features()

# Ensure input_df columns match model feature order
input_df = input_df[feature_columns]

# Processing Button
if st.button("Predict"):
    if rf_model:
        try:
            # Prediction
            rf_proba = rf_model.predict_proba(input_df)[:, 1]
            st.write(f"**Random Forest Prediction: CVD Risk Probability = {rf_proba[0]:.2f}**")

            # Prediction Probability Distribution
            st.subheader("Prediction Probability Distribution")
            fig, ax = plt.subplots()
            bar = ax.barh(["Random Forest"], [rf_proba[0]], color="blue")
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

            # Feature Importances (Random Forest)
            st.subheader("Feature Importances (Random Forest)")
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importances[indices], color='blue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_columns[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importances (Random Forest)')
            st.pyplot(fig)

            # Notes
            st.subheader("Notes")
            st.write("""
                - These predictions are for informational purposes only.
                - Consult a healthcare professional for medical advice.
                - The model uses a Random Forest approach with multiple features.
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing predictions or plotting: {e}")
    else:
        st.error("Model not loaded successfully.")
