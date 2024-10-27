# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the complete stacking model
@st.cache_resource
def load_stacking_model():
    try:
        stacking_model = joblib.load('genai_stacking_model.pkl')
        return stacking_model
    except Exception as e:
        st.error(f"Error loading the stacking model: {e}")
        return None

stacking_model = load_stacking_model()

# Sidebar inputs for user parameters
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
    prevhyp = st.sidebar.selectbox('Previous Hypertension:', (0, 1))
    ldlc = st.sidebar.slider('LDL Cholesterol:', 20, 565, 100)
    hdlc = st.sidebar.slider('HDL Cholesterol:', 10, 189, 50)

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
        'PREVHYP': prevhyp,
        'LDLC': ldlc,
        'HDLC': hdlc
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display predictions on clicking Predict button
if st.sidebar.button('Predict') and stacking_model is not None:
    try:
        # Predict using stacking model
        stacking_proba = stacking_model.predict_proba(input_df)
        st.subheader('Predictions')
        st.write(f"Stacking Model Prediction: CVD probability {stacking_proba[0, 1]:.2f}")
        
        # Display Probability Distribution
        st.subheader('Prediction Probability Distribution')
        fig, ax = plt.subplots()
        ax.bar(['Stacking Model'], [stacking_proba[0, 1]], color='blue')
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error making predictions: {e}")

    # Feature Importances from XGBoost
    st.subheader('Feature Importances (XGBoost)')
    try:
        xgb_model = stacking_model.named_estimators_['xgb']  # Access the XGBoost model
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots()
        ax.barh(range(len(indices)), importances[indices], color='blue', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([input_df.columns[i] for i in indices])
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting feature importances: {e}")

    # Model Performance (ROC Curve)
    st.subheader('Model Performance')
    try:
        fig, ax = plt.subplots()
        # Assuming the full data and labels are available as `X_full` and `y_full`
        # fpr, tpr, _ = roc_curve(y_full, stacking_model.predict_proba(X_full)[:, 1])
        # ax.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc_score(y_full, stacking_model.predict_proba(X_full)[:, 1]):.2f})')
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
