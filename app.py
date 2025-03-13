import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

# Set page config for a wider layout
st.set_page_config(page_title="CVD Risk Prediction", layout="wide")

# Function to download a file if it doesn’t exist
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# URLs for model files
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/stacking_genai_model.pkl'
scaler_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/scaler.pkl'

# Local paths
stacking_model_path = 'stacking_genai_model.pkl'
scaler_path = 'scaler.pkl'

# Download models and scaler if not present
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
        loaded_object = joblib.load(stacking_model_path)
        if isinstance(loaded_object, dict) and 'gen_stacking_meta_model' in loaded_object:
            return {
                'meta_model': loaded_object['gen_stacking_meta_model'],
                'base_models': {
                    'rf': loaded_object.get('rf_model'),
                    'xgb': loaded_object.get('xgb_model')
                }
            }
        else:
            st.error("Model structure incorrect. Please check model file.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()

# Load the scaler
scaler = joblib.load(scaler_path)

# Define feature columns
feature_columns = [
    'SEX', 'AGE', 'educ', 'CURSMOKE', 'CIGPDAY', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE',
    'GLUCOSE', 'HDLC', 'LDLC', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
]

# Title
st.title("🫀 Cardiovascular Disease (CVD) Risk Prediction")
st.write("This tool helps assess your potential risk of developing CVD based on clinical parameters.")

# Sidebar Inputs
st.sidebar.header("📋 Enter Your Health Details")

user_data = {feature: st.sidebar.slider(feature, float(30), float(250), float(100)) if feature not in ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP']
             else st.sidebar.selectbox(feature, [0, 1]) for feature in feature_columns}

input_df = pd.DataFrame([user_data])

# Scale input data
input_df_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔍 Predict Risk"):
    if stacking_model:
        try:
            # Base model predictions
            meta_features = []
            for model_name, base_model in stacking_model['base_models'].items():
                if hasattr(base_model, 'predict_proba'):
                    proba = base_model.predict_proba(input_df_scaled.reshape(1, -1))[:, 1]
                else:
                    proba = base_model.predict(input_df_scaled.reshape(1, -1))
                meta_features.append(proba)

            # Stack base model predictions
            meta_input = np.column_stack(meta_features)

            # Final meta-model prediction
            meta_proba = stacking_model['meta_model'].predict_proba(meta_input)[:, 1][0]

            # 🏥 Risk Level Classification
            if meta_proba < 0.3:
                risk_level = "🟢 Low Risk"
                risk_description = "Your CVD risk is low. Maintain a healthy lifestyle to keep it that way!"
            elif 0.3 <= meta_proba < 0.7:
                risk_level = "🟡 Moderate Risk"
                risk_description = "You have a moderate risk of CVD. Consider making lifestyle improvements."
            else:
                risk_level = "🔴 High Risk"
                risk_description = "Your CVD risk is high. It is recommended to consult with a doctor for further evaluation."

            # Display results
            st.subheader("🧪 Prediction Results")
            st.metric(label="**CVD Risk Probability**", value=f"{meta_proba:.2%}", delta_color="inverse")
            st.success(f"**Risk Level: {risk_level}**")
            st.write(risk_description)

            # 📊 Probability Distribution
            st.subheader("📈 Probability Distribution")
            fig, ax = plt.subplots(figsize=(8, 1))
            color = "green" if meta_proba < 0.3 else "yellow" if meta_proba < 0.7 else "red"
            ax.barh(["CVD Risk"], [meta_proba], color=color)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            for rect in ax.patches:
                width = rect.get_width()
                ax.text(width + 0.02, rect.get_y() + rect.get_height()/2, f"{width*100:.0f}%", va="center")
            st.pyplot(fig)

            # 🔬 Feature Importance (XGBoost)
            st.subheader("🔎 Feature Importance (XGBoost)")
            xgb_model = stacking_model['base_models']['xgb']
            if hasattr(xgb_model, 'feature_importances_'):
                importances = xgb_model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(np.array(feature_columns)[sorted_indices], importances[sorted_indices], color="blue")
                ax.set_xlabel("Feature Importance")
                ax.invert_yaxis()
                st.pyplot(fig)

            # 📉 Model Performance
            st.subheader("📊 Model Performance")
            st.write("This model has been evaluated on test data with an **AUC of 0.96**, ensuring high reliability in CVD risk assessment.")

            # ℹ️ Notes
            st.subheader("ℹ️ Important Notes")
            st.info("""
                - This tool provides **CVD risk estimation** based on medical data.
                - Predictions are **not a substitute for professional medical advice**.
                - For **high-risk results**, it is strongly recommended to **consult with a physician**.
                - A healthy lifestyle including **diet, exercise, and regular medical checkups** can reduce cardiovascular risks.
            """)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.error("⚠️ Model loading failed. Please check the model file.")

# Footer
st.write("---")
st.write("Developed by **Howard Nguyen** | Data Science & AI | 2025")
