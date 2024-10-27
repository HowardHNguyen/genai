import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.models import load_model
import urllib.request
import os

# URLs for the model files in your GitHub repository
rf_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/rf_model.pkl'
xgbm_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/xgbm_model.pkl'
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/genai_stacking_model.pkl'
cnn_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/cnn_model.h5'

# Local paths for the model files
rf_model_path = 'rf_model.pkl'
xgbm_model_path = 'xgbm_model.pkl'
stacking_model_path = 'genai_stacking_model.pkl'
cnn_model_path = 'cnn_model.h5'

# Function to download a file from a URL
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# Download model files if they are not already present
if not os.path.exists(rf_model_path):
    st.info(f"Downloading {rf_model_path}...")
    download_file(rf_model_url, rf_model_path)

if not os.path.exists(xgbm_model_path):
    st.info(f"Downloading {xgbm_model_path}...")
    download_file(xgbm_model_url, xgbm_model_path)

if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

if not os.path.exists(cnn_model_path):
    st.info(f"Downloading {cnn_model_path}...")
    download_file(cnn_model_url, cnn_model_path)

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    try:
        rf_model = joblib.load(rf_model_path)
        xgbm_model = joblib.load(xgbm_model_path)
        stacking_model = joblib.load(stacking_model_path)
        cnn_model = load_model(cnn_model_path)
        return rf_model, xgbm_model, stacking_model, cnn_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

rf_model, xgbm_model, stacking_model, cnn_model = load_models()

# Define input parameters for user
def user_input_features():
    age = st.sidebar.slider('Enter your age:', 32, 81, 54)
    totchol = st.sidebar.slider('Total Cholesterol:', 107, 696, 200)
    sysbp = st.sidebar.slider('Systolic Blood Pressure:', 83, 295, 151)
    diabp = st.sidebar.slider('Diastolic Blood Pressure:', 30, 150, 89)
    bmi = st.sidebar.slider('BMI:', 14.43, 56.80, 26.77)
    cursmoke = st.sidebar.selectbox('Current Smoker:', (0, 1))
    glucose = st.sidebar.slider('Glucose:', 39, 478, 117)
    diabetes = st.sidebar.selectbox('Diabetes:', (0, 1))
    heartrate = st.sidebar.slider('Heart Rate:', 37, 220, 91)
    cigpday = st.sidebar.slider('Cigarettes Per Day:', 0, 90, 20)
    bpmeds = st.sidebar.selectbox('On BP Meds:', (0, 1))
    stroke = st.sidebar.selectbox('Stroke:', (0, 1))
    hyperten = st.sidebar.selectbox('Hypertension:', (0, 1))

    data = {
        'AGE': age,
        'TOTCHOL': totchol,
        'SYSBP': sysbp,
        'DIABP': diabp,
        'BMI': bmi,
        'CURSMOKE': cursmoke,
        'GLUCOSE': glucose,
        'DIABETES': diabetes,
        'HEARTRTE': heartrate,
        'CIGPDAY': cigpday,
        'BPMEDS': bpmeds,
        'STROKE': stroke,
        'HYPERTEN': hyperten
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
if st.sidebar.button('PREDICT NOW'):
    if rf_model and xgbm_model and stacking_model and cnn_model:
        try:
            # Predict probabilities with the stacking model
            stacking_proba = stacking_model.predict_proba(input_df)[:, 1]
            
            # Predict with the CNN model (dummy prediction, assuming input shape matches)
            cnn_input = np.array(input_df).reshape(1, -1, 1)  # Adjust input shape if necessary
            cnn_proba = cnn_model.predict(cnn_input)[0, 1]

            # Display predictions
            st.write(f"Stacking Model Prediction: CVD probability {stacking_proba[0]:.2f}")
            st.write(f"CNN Model Prediction: CVD probability {cnn_proba:.2f}")
        
            # Probability distribution plot
            st.subheader('Prediction Probability Distribution')
            fig, ax = plt.subplots()
            bars = ax.bar(['Stacking Model', 'CNN Model'], [stacking_proba[0], cnn_proba], color=['blue', 'orange'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom')
            st.pyplot(fig)
        
            # Feature Importances (Random Forest)
            st.subheader('Feature Importances (Random Forest)')
            rf_importances = rf_model.feature_importances_
            fig, ax = plt.subplots()
            indices = np.argsort(rf_importances)
            ax.barh(range(len(indices)), rf_importances[indices], align='center', color='blue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([input_df.columns[i] for i in indices])
            ax.set_xlabel('Importance')
            st.pyplot(fig)

            # ROC Curve for Stacking Model
            st.subheader('Model Performance')
            fig, ax = plt.subplots()
            fpr, tpr, _ = roc_curve([0, 1], stacking_proba)  # Placeholder for true label; replace as needed
            ax.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc_score([0, 1], stacking_proba):.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='best')
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    else:
        st.error("Models could not be loaded properly.")
else:
    st.write("## Cardiovascular Disease Prediction App")
    st.write("### Enter your parameters and click Predict to get the results.")
