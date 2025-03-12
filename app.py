import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
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

# Local paths for models
stacking_model_path = 'stacking_genai_model.pkl'

# Download models if they don’t exist
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

# Load the stacking model
@st.cache(allow_output_mutation=True)
def load_stacking_model():
    try:
        # Load the content of the .pkl file
        loaded_object = joblib.load(stacking_model_path)
        if isinstance(loaded_object, dict):
            if 'gen_stacking_meta_model' in loaded_object and hasattr(loaded_object['gen_stacking_meta_model'], 'predict_proba'):
                meta_model = loaded_object['gen_stacking_meta_model']
                # Extract base models
                base_models = {
                    'rf': loaded_object.get('rf_model'),
                    'xgb': loaded_object.get('xgb_model'),
                }
                return {'meta_model': meta_model, 'base_models': base_models}
            else:
                st.error("No 'gen_stacking_meta_model' found or it doesn’t support 'predict_proba'.")
                return None
        else:
            st.error(f"Loaded object is of type {type(loaded_object)} and not a dictionary.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()

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
if st.button("PREDICT"):
    if stacking_model is None or 'meta_model' not in stacking_model or 'base_models' not in stacking_model:
        st.error("Cannot make predictions: Model or base models failed to load.")
    else:
        try:
            # Generate predictions from base models
            meta_features = []
            for model_name, base_model in stacking_model['base_models'].items():
                if base_model is not None and hasattr(base_model, 'predict_proba'):
                    proba = base_model.predict_proba(input_df)[:, 1]  # Probability of positive class
                    meta_features.append(proba)
                else:
                    st.error(f"Base model {model_name} is None or does not support predict_proba.")
                    raise Exception("Invalid base model.")

            # Combine into a single input for the meta-model
            meta_input = np.column_stack(meta_features)

            # Prediction using meta-model
            meta_proba = stacking_model['meta_model'].predict_proba(meta_input)[:, 1]
            st.write(f"**Stacking Model Prediction: CVD Risk Probability = {meta_proba[0]:.2f}**")

            # Prediction Probability Distribution
            st.subheader("Prediction Probability Distribution")
            fig, ax = plt.subplots()
            ax.barh(["Stacking Model"], [meta_proba[0]], color="red")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

            # Feature Importance Plot using Random Forest
            st.subheader("Feature Importance / Risk Factors (Random Forest)")
            rf_model = stacking_model['base_models']['rf']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                indices = np.argsort(importances)[::-1]  
                fig2, ax2 = plt.subplots()
                ax2.barh([feature_columns[i] for i in indices], importances[indices], color="green")
                ax2.set_xlabel("Importance")
                ax2.invert_yaxis()
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error processing predictions: {e}")
