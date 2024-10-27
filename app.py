import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import urllib.request

# Function to download the file
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# URLs for the model files
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/stacking_model.pkl'

# Local paths for the model files
stacking_model_path = 'stacking_model.pkl'

# Download the models if not already present
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

# Load the stacking model
try:
    stacking_model = joblib.load(stacking_model_path)
    if not hasattr(stacking_model, 'predict_proba'):
        st.error("Loaded model does not have the required methods. Please check the model file.")
except Exception as e:
    st.error(f"Error loading models: {e}")
    stacking_model = None

# Load the dataset
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd/master/frmgham2.csv'
try:
    data = pd.read_csv(data_url)
    data.fillna(data.mean(), inplace=True)  # Handle missing values
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = None

# Define the feature columns
feature_columns = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 'GLUCOSE', 'DIABETES', 
                   'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN']

# Sidebar for input parameters
st.sidebar.header('Enter your parameters')
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
        'AGE': age, 'TOTCHOL': totchol, 'SYSBP': sysbp, 'DIABP': diabp, 'BMI': bmi, 
        'CURSMOKE': cursmoke, 'GLUCOSE': glucose, 'DIABETES': diabetes, 'HEARTRTE': heartrate, 
        'CIGPDAY': cigpday, 'BPMEDS': bpmeds, 'STROKE': stroke, 'HYPERTEN': hyperten
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Apply the model to make predictions
if stacking_model and st.sidebar.button('PREDICT NOW'):
    try:
        stacking_proba = stacking_model.predict_proba(input_df)[:, 1]
        st.write(f"Stacking Model Prediction: CVD with probability {stacking_proba[0]:.2f}")
    except Exception as e:
        st.error(f"Error making predictions: {e}")

    # Prediction Probability Distribution
    st.subheader('Prediction Probability Distribution')
    try:
        fig, ax = plt.subplots()
        ax.bar(['Stacking Model'], [stacking_proba[0]], color=['blue'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.text(0, stacking_proba[0], f'{stacking_proba[0]:.2f}', ha='center', va='bottom')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting probability distribution: {e}")

    # Feature Importances (Random Forest)
    st.subheader('Feature Importances (Random Forest)')
    try:
        rf_model = stacking_model.named_estimators_.get('rf')
        if rf_model:
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)
            fig, ax = plt.subplots()
            ax.barh(range(len(indices)), importances[indices], color='blue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_columns[i] for i in indices])
            ax.set_xlabel('Importance')
            st.pyplot(fig)
        else:
            st.error("Random Forest model not found in the stacking model.")
    except Exception as e:
        st.error(f"Error plotting feature importances: {e}")

    # ROC Curve
    st.subheader('Model Performance')
    try:
        fpr, tpr, _ = roc_curve(data['CVD'], stacking_model.predict_proba(data[feature_columns])[:, 1])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc_score(data["CVD"], stacking_model.predict_proba(data[feature_columns])[:, 1]):.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='best')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting ROC curve: {e}")
else:
    st.write("## Cardiovascular Disease Prediction App")
    st.write("### Enter your parameters and click Predict to get the results.")
