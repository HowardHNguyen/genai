import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from scikeras.wrappers import KerasClassifier
import os
import urllib.request

# Define the CNN model architecture function
def create_cnn_model(input_shape=(16, 1)):
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to download the file
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# URLs for the model files
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/genai_stacking_model.pkl'
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/frmgham2.csv'

# Local paths for the model files
stacking_model_path = 'genai_stacking_model.pkl'

# Download the models if not already present
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

# Load the stacking model
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_stacking_model():
    try:
        model = joblib.load(stacking_model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

stacking_model, loading_error = load_stacking_model()
if loading_error:
    st.error(loading_error)

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    try:
        data = pd.read_csv(data_url)
        data.fillna(data.mean(), inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

# Define the feature columns
feature_columns = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE',
                   'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS',
                   'STROKE', 'HYPERTEN', 'PREVHYP', 'LDLC', 'HDLC']

# Sidebar for input parameters
st.sidebar.header('Enter your parameters')

def user_input_features():
    stroke = st.sidebar.selectbox('Stroke:', (0, 1))
    sysbp = st.sidebar.slider('Systolic Blood Pressure:', 83, 295, 151)
    age = st.sidebar.slider('Enter your age:', 32, 81, 54)
    prevhyp = st.sidebar.selectbox('Previous Hypertension:', (0, 1))
    hyperten = st.sidebar.selectbox('Hypertension:', (0, 1))
    diabp = st.sidebar.slider('Diastolic Blood Pressure:', 30, 150, 89)
    diabetes = st.sidebar.selectbox('Diabetes:', (0, 1))
    bpmeds = st.sidebar.selectbox('On BP Meds:', (0, 1))
    bmi = st.sidebar.slider('BMI:', 14.43, 56.80, 26.77)
    glucose = st.sidebar.slider('Glucose:', 39, 478, 117)
    totchol = st.sidebar.slider('Total Cholesterol:', 107, 696, 200)
    cigpday = st.sidebar.slider('Cigarettes Per Day:', 0, 90, 20)
    ldlc = st.sidebar.slider('LDLC:', 10, 189, 130)
    cursmoke = st.sidebar.selectbox('Current Smoker:', (0, 1))
    heartrate = st.sidebar.slider('Heart Rate:', 37, 220, 91)
    hdlc = st.sidebar.slider('HDLC:', 20, 565, 50)

    data = {
        'STROKE': stroke,
        'SYSBP': sysbp,
        'AGE': age,
        'PREVHYP': prevhyp,
        'HYPERTEN': hyperten,
        'DIABP': diabp,
        'DIABETES': diabetes,
        'BPMEDS': bpmeds,
        'BMI': bmi,
        'GLUCOSE': glucose,
        'TOTCHOL': totchol,
        'CIGPDAY': cigpday,
        'LDLC': ldlc,        
        'CURSMOKE': cursmoke,
        'HEARTRTE': heartrate,
        'HDLC': hdlc
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure input_df columns match feature_columns order
input_df = input_df[feature_columns]

# Apply the model to make predictions
if st.sidebar.button('Predict'):
    if stacking_model is not None:
        try:
            stacking_proba = stacking_model.predict_proba(input_df)[:, 1]
            st.write("## Cardiovascular Disease Prediction App")
            st.subheader('Predictions')
            st.write(f"Stacking Model Prediction: CVD with probability {stacking_proba[0]:.2f}")
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    else:
        st.error("Model could not be loaded.")

    # Plot prediction probability distribution
    st.subheader('Prediction Probability Distribution')
    try:
        fig, ax = plt.subplots()
        ax.bar(['Stacking Model'], [stacking_proba[0]], color=['blue'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting probability distribution: {e}")

    # Plot feature importances for XGBoost
    st.subheader('Feature Importances (XGBoost)')
    try:
        xgb_model = stacking_model.named_estimators_['xgb']  # Access XGBoost model in stack
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

    # Plot ROC curve for the model
    st.subheader('Model Performance')
    try:
        fpr, tpr, _ = roc_curve(data['CVD'], stacking_model.predict_proba(data[feature_columns])[:, 1])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc_score(data["CVD"], stacking_model.predict_proba(data[feature_columns])[:, 1]):.2f})')
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
