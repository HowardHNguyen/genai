import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
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
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/master/genai_stacking_model.pkl'
cnn_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/master/cnn_model.h5'

# Local paths for the model files
stacking_model_path = 'genai_stacking_model.pkl'
cnn_model_path = 'cnn_model.h5'

# Download the models if not already present
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

if not os.path.exists(cnn_model_path):
    st.info(f"Downloading {cnn_model_path}...")
    download_file(cnn_model_url, cnn_model_path)

# Load the stacking model and CNN model
try:
    stacking_model = joblib.load(stacking_model_path)
    cnn_model = load_model(cnn_model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")

# Load the dataset
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/master/frmgham2.csv'
try:
    data = pd.read_csv(data_url)
except Exception as e:
    st.error(f"Error loading data: {e}")

# Handle missing values by replacing them with the mean of the respective columns
if 'data' in locals():
    data.dropna(inplace=True)
    # data.fillna(data.mean(), inplace=True)

# Define the feature columns
feature_columns = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE',
                   'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS',
                   'STROKE', 'HYPERTEN', 'LDLC', 'HDLC']

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
    ldlc = st.sidebar.slider('LDLC:', 20, 565, 150)
    hdlc = st.sidebar.slider('HDLC:', 10, 189, 55)

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
        'HYPERTEN': hyperten,
        'LDLC': ldlc,
        'HDLC': hdlc
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Apply the model to make predictions
if st.sidebar.button('Predict'):
    try:
        # Get prediction probabilities from the stacking model
        stacking_proba = stacking_model.predict_proba(input_df)[:, 1]
    except Exception as e:
        st.error(f"Error making predictions: {e}")

    st.write("""
    # Cardiovascular Disease Prediction App
    This app predicts the probability of cardiovascular disease (CVD) using user inputs.
    """)

    st.subheader('Predictions')
    try:
        st.write(f"Stacking Model Prediction: CVD with probability {stacking_proba[0]:.2f}")
    except:
        pass

    # Plot the prediction probability distribution
    st.subheader('Prediction Probability Distribution')
    try:
        fig, ax = plt.subplots()
        bars = ax.bar(['Stacking Model'], [stacking_proba[0]], color=['blue'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom')  # va: vertical alignment
        st.pyplot(fig)
    except:
        pass

    # Plot feature importances for Random Forest
    st.subheader('Feature Importances (Random Forest)')
    try:
        rf_base_model = stacking_model.named_estimators_['rf']  # Assuming 'rf' is the RandomForest key in your stacking model
        fig, ax = plt.subplots()
        importances = rf_base_model.feature_importances_
        indices = np.argsort(importances)
        ax.barh(range(len(indices)), importances[indices], color='blue', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting feature importances: {e}")

    # Plot ROC curve for the stacking model
    st.subheader('Model Performance')
    try:
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(data['CVD'], stacking_model.predict_proba(data[feature_columns])[:, 1])
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
    st.write("## Cardiovascular Disease Prediction by Gen AI, Howard Nguyen")
    st.write("### Enter your parameters and click Predict to get the results.")
