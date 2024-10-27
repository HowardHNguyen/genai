import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import urllib.request

# Define feature columns in the correct training order
feature_columns = ['STROKE', 'SYSBP', 'AGE', 'PREVHYP', 'HYPERTEN', 'DIABP', 'DIABETES', 'BPMEDS', 
                   'BMI', 'GLUCOSE', 'TOTCHOL', 'CIGPDAY', 'LDLC', 'CURSMOKE', 'HEARTRTE', 'HDLC']

# URL to download the stacking model
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/genai_stacking_model.pkl'
stacking_model_path = 'genai_stacking_model.pkl'

# Function to download the model file if not present
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# Download the model if not already present
if not os.path.exists(stacking_model_path):
    download_file(stacking_model_url, stacking_model_path)

# Load the stacking model
@st.cache(allow_output_mutation=True)
def load_stacking_model():
    try:
        model = joblib.load(stacking_model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()

# Sidebar for input parameters
st.sidebar.header('Enter your parameters')
def user_input_features():
    # Define all input fields as before
    data = {
        'STROKE': st.sidebar.selectbox('Stroke:', (0, 1)),
        'SYSBP': st.sidebar.slider('Systolic Blood Pressure:', 83, 295, 151),
        'AGE': st.sidebar.slider('Enter your age:', 32, 81, 54),
        'PREVHYP': st.sidebar.selectbox('Previous Hypertension:', (0, 1)),
        'HYPERTEN': st.sidebar.selectbox('Hypertension:', (0, 1)),
        'DIABP': st.sidebar.slider('Diastolic Blood Pressure:', 30, 150, 89),
        'DIABETES': st.sidebar.selectbox('Diabetes:', (0, 1)),
        'BPMEDS': st.sidebar.selectbox('On BP Meds:', (0, 1)),
        'BMI': st.sidebar.slider('BMI:', 14.43, 56.80, 26.77),
        'GLUCOSE': st.sidebar.slider('Glucose:', 39, 478, 117),
        'TOTCHOL': st.sidebar.slider('Total Cholesterol:', 107, 696, 200),
        'CIGPDAY': st.sidebar.slider('Cigarettes Per Day:', 0, 90, 20),
        'LDLC': st.sidebar.slider('LDLC:', 10, 189, 130),
        'CURSMOKE': st.sidebar.selectbox('Current Smoker:', (0, 1)),
        'HEARTRTE': st.sidebar.slider('Heart Rate:', 37, 220, 91),
        'HDLC': st.sidebar.slider('HDLC:', 20, 565, 50)
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure input columns match feature_columns order
input_df = input_df[feature_columns]

# Apply the model to make predictions
if st.sidebar.button('Predict'):
    if stacking_model:
        try:
            # Adjust input for model compatibility
            if 'cnn' in [name for name, _ in stacking_model.estimators]:
                # CNN expects 3D shape for each sample; reshape accordingly
                input_data = np.expand_dims(input_df.values, axis=2)
            else:
                input_data = input_df.values

            # Make prediction
            stacking_proba = stacking_model.predict_proba(input_data)[:, 1]
            st.subheader('Predictions')
            st.write(f"Stacking Model Prediction: CVD with probability {stacking_proba[0]:.2f}")

            # Plot probability distribution
            st.subheader('Prediction Probability Distribution')
            fig, ax = plt.subplots()
            ax.bar(['Stacking Model'], [stacking_proba[0]], color='blue')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            st.pyplot(fig)

            # Feature importances (if available)
            st.subheader('Feature Importances (XGBoost)')
            try:
                xgb_model = stacking_model.named_estimators_['xgb']
                importances = xgb_model.feature_importances_
                fig, ax = plt.subplots()
                indices = np.argsort(importances)
                ax.barh(range(len(indices)), importances[indices], color='blue')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_columns[i] for i in indices])
                ax.set_xlabel('Importance')
                st.pyplot(fig)
            except AttributeError:
                st.write("Feature importances not available for XGBoost.")

            # ROC Curve
            st.subheader('Model Performance')
            fpr, tpr, _ = roc_curve(data['CVD'], stacking_model.predict_proba(data[feature_columns])[:, 1])
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc_score(data["CVD"], stacking_model.predict_proba(data[feature_columns])[:, 1]):.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    else:
        st.error("Model could not be loaded.")
else:
    st.write("## Cardiovascular Disease Prediction App")
    st.write("Enter your parameters and click Predict to get the results.")
