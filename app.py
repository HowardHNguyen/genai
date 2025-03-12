import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import urllib.request

# Function to download a file if it doesn't exist
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")

# URLs for model files on GitHub
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/stacking_genai_model.pkl'
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/frmgham2.csv'

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
        model = joblib.load("stacking_genai_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()

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

# Define the feature columns (ensure they match model training order)
feature_columns = [
    'SEX', 'AGE', 'educ', 'CURSMOKE', 'CIGPDAY', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE',
    'GLUCOSE', 'HDLC', 'LDLC', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
]

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

# Apply model predictions
if st.sidebar.button('PREDICT'):
    if stacking_model:
        try:
            rf_proba = stacking_model['rf_model'].predict_proba(input_df)[:, 1]
            xgb_proba = stacking_model['xgb_model'].predict_proba(input_df)[:, 1]

            # Combine predictions into meta-model input
            meta_input = np.column_stack([rf_proba, xgb_proba])
            stacking_proba = stacking_model['meta_model'].predict_proba(meta_input)[:, 1]

            st.subheader('Predictions')
            st.write(f"Stacking Model Prediction: CVD with probability {stacking_proba[0]:.2f}")

            # Plot prediction probability distribution
            fig, ax = plt.subplots()
            ax.bar(['Stacking Model'], [stacking_proba[0]], color='blue')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    else:
        st.error("Model could not be loaded.")

    # Feature Importance (XGBoost)
    st.subheader('Feature Importances (XGBoost)')
    try:
        xgb_model = stacking_model['xgb_model']
        importances = xgb_model.feature_importances_
        fig, ax = plt.subplots()
        indices = np.argsort(importances)
        ax.barh(range(len(indices)), importances[indices], color='blue', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting feature importances: {e}")

    # ROC Curve
    st.subheader('Model Performance')
    try:
        fpr, tpr, _ = roc_curve(data['CVD'], stacking_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc_score(data["CVD"], stacking_proba):.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='best')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting ROC curve: {e}")

else:
    st.write("## Cardiovascular Disease Prediction App")
    st.write("### Enter your parameters and click Predict to get the results.")
