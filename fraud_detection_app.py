# fraud_detection_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Churn_Modelling.csv")
    return data

# Preprocess data
def preprocess_data(df):
    df = df.select_dtypes(include=[np.number])  # Keep numeric columns
    if 'Exited' in df.columns:
        y = df['Exited']  # Assuming 'Exited' is the target (adjust if needed)
        X = df.drop('Exited', axis=1)
    else:
        st.error("Target column 'Exited' not found.")
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Streamlit App
st.title("Credit Card Fraud Detection App")
st.write("Upload your dataset and detect potential frauds using Machine Learning.")

data = load_data()
st.write("### Dataset Preview", data.head())

X, y = preprocess_data(data)
if X is not None and y is not None:
    model, X_test, y_test = train_model(X, y)
    y_pred = model.predict(X_test)

    st.write("### Model Evaluation")
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("### Make a Prediction")
    input_data = st.text_input("Enter comma-separated values for a new record:")

    if st.button("Predict"):
        try:
            input_array = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
            input_scaled = StandardScaler().fit_transform(X)  # Fit on entire data for simplicity
            input_transformed = StandardScaler().fit(X).transform(input_array)
            prediction = model.predict(input_transformed)
            st.success(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
