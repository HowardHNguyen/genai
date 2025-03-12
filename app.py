import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from sklearn.preprocessing import StandardScaler

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
scaler_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/scaler.pkl'

# Local paths for models
stacking_model_path = 'stacking_genai_model.pkl'
scaler_path = 'scaler.pkl'

# Download models and scaler if they don’t exist
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

if not os.path.exists(scaler_path):
    st.info(f"Downloading {scaler_path}...")
    download_file(scaler_url, scaler_path)

# Load the stacking model
@st.cache_resource
def load_stacking_model():
    try:
        # Load the content of the .pkl file
        loaded_object = joblib.load(stacking_model_path)
        if isinstance(loaded_object, dict):
            if 'gen_stacking_meta_model' in loaded_object and hasattr(loaded_object['gen_stacking_meta_model'], 'predict_proba'):
                meta_model = loaded_object['gen_stacking_meta_model']
                # Extract base models (excluding CNN)
                base_models = {
                    'rf': loaded_object.get('rf_model'),
                    'xgb': loaded_object.get('xgb_model')
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

# Load the scaler
scaler = joblib.load(scaler_path)

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
age = st.sidebar.slider("AGE", 32.0, 81.0, 32.0)  # Default to low-risk
educ = st.sidebar.slider("Education Level (educ)", 1.0, 4.0, 1.99)
cursmoke = st.sidebar.selectbox("Current Smoker (0 = No, 1 = Yes)", [0, 1], index=0)  # Default to No
cigpday = st.sidebar.slider("Cigarettes per Day", 0.0, 90.0, 0.0)
totchol = st.sidebar.slider("Total Cholesterol", 107.0, 696.0, 150.0)  # Lower risk
sysbp = st.sidebar.slider("Systolic BP", 83.5, 295.0, 90.0)  # Lower risk
diabp = st.sidebar.slider("Diastolic BP", 30.0, 159.0, 60.0)  # Lower risk
bmi = st.sidebar.slider("BMI", 15.0, 59.0, 20.0)  # Healthy
heartrte = st.sidebar.slider("Heart Rate", 40.0, 120.0, 60.0)  # Lower risk
glucose = st.sidebar.slider("Glucose", 50.0, 360.0, 80.0)
hdlc = st.sidebar.slider("HDL Cholesterol", 20.0, 100.0, 60.0)  # Higher good cholesterol
ldlc = st.sidebar.slider("LDL Cholesterol", 20.0, 300.0, 176.47)  # Mean value
diabetes = st.sidebar.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1], index=0)
bpmeds = st.sidebar.selectbox("BP Meds (0 = No, 1 = Yes)", [0, 1], index=0)
prevchd = st.sidebar.selectbox("Prev CHD (0 = No, 1 = Yes)", [0, 1], index=0)
prevap = st.sidebar.selectbox("PREVAP (0 = No, 1 = Yes)", [0, 1], index=0)
prevmi = st.sidebar.selectbox("PREMI (0 = No, 1 = Yes)", [0, 1], index=0)
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

# Scale input data using the loaded scaler
input_df_scaled = scaler.transform(input_df)
st.write("Scaled Input Values:", input_df_scaled[0])  # Debug: Show scaled input

# Processing Button
if st.button("Predict"):
    if stacking_model is None or 'meta_model' not in stacking_model or 'base_models' not in stacking_model:
        st.error("Cannot make predictions: Model or base models failed to load.")
    else:
        try:
            # Generate predictions from base models (RF and XGB only)
            meta_features = []
            for model_name, base_model in stacking_model['base_models'].items():
                if base_model is not None:
                    if hasattr(base_model, 'predict_proba'):
                        proba = base_model.predict_proba(input_df_scaled.reshape(1, -1))[:, 1]  # Probability of positive class
                        st.write(f"{model_name.upper()} Prediction: {proba[0]}")  # Debug
                    else:
                        proba = base_model.predict(input_df_scaled.reshape(1, -1))  # Fallback to predict if no predict_proba
                        st.write(f"{model_name.upper()} Prediction: {proba[0]}")  # Debug
                    meta_features.append(proba)
                else:
                    st.error(f"Base model {model_name} is None.")
                    raise Exception("Invalid base model.")

            # Combine into a single input for the meta-model (should be 2 features)
            meta_input = np.column_stack(meta_features)
            st.write(f"Meta-input: {meta_input}")  # Debug

            # Ensure meta-input has 2 features
            if meta_input.shape[1] != 2:
                st.error(f"Meta-input has {meta_input.shape[1]} features, but meta-model expects 2. Check base models.")
                raise Exception("Feature mismatch.")

            # Prediction using meta-model
            meta_proba = stacking_model['meta_model'].predict_proba(meta_input)[:, 1]
            st.write(f"**Stacking Model Prediction: CVD Risk Probability = {meta_proba[0]:.2f}**")

            # Prediction Probability Distribution (Red Color, Increased Height)
            st.subheader("Prediction Probability Distribution")
            fig, ax = plt.subplots(figsize=(8, 0.75))  # Increased height to 0.75
            bar = ax.barh(["Stacking Model"], [meta_proba[0]], color="red")  # Changed to red
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            # Add percentage label to the bar
            for rect in bar:
                width = rect.get_width()
                ax.text(width + 0.01, rect.get_y() + rect.get_height()/2, f"{width*100:.0f}%", va="center")
            st.pyplot(fig)

            # Feature Importance / Risk Factor Plot
            st.subheader("Feature Importance / Risk Factors (Random Forest)")
            # Use Random Forest model for feature importance
            rf_model = stacking_model['base_models']['rf']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                indices = np.argsort(importances)[::-1]  # Sort by importance
                top_n = 10  # Show top 10 features
                top_indices = indices[:top_n]
                top_importances = importances[top_indices]
                top_features = [feature_columns[i] for i in top_indices]

                # Plot feature importance
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.barh(top_features, top_importances, color="green")
                ax2.set_xlabel("Importance")
                ax2.invert_yaxis()  # Highest importance at the top
                st.pyplot(fig2)
            else:
                st.warning("Feature importance not available for Random Forest model.")

            # Model Performance
            st.subheader("Model Performance")
            st.write("The model has been evaluated on a test dataset with an AUC of 0.96.")

            # Notes
            st.subheader("Notes")
            st.write("""
                - These predictions are for informational purposes only.
                - Consult a healthcare professional for medical advice.
                - The model uses a stacking approach with Random Forest and XGBoost only (CNN excluded).
            """, unsafe_allow_html=True)

        except AttributeError as e:
            st.error(f"Model error: {e}. Check if base models support predict_proba or predict.")
        except Exception as e:
            st.error(f"Error processing predictions or plotting: {e}")