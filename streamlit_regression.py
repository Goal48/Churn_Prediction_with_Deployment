import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load the scaler & One Hot Encoder
with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open ('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
with open ('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo = pickle.load(file)

# Streamlit app
st.title("Salary Regression Prediction")
# Input fields
credit_score = st.number_input("Credit Score", min_value=0, max_value=850, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, step=1)
tenure = st.number_input("Tenure", min_value=0, step=1)
balance = st.number_input("Balance", min_value=0.0, step=0.01)
num_of_products = st.number_input("Number of Products", min_value=0, step=1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])   
exited = st.selectbox("Exited", ["Yes", "No"])
# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'Geography': [geography],
    'CreditScore': [credit_score],
    'Exited': [1 if exited == "Yes" else 0]
})
# One-hot encode the 'Geography' column
geography_encoded = one_hot_encoder_geo.transform(input_data[['Geography']]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geography_encoded_df], axis=1)

# Drop the original 'Geography' column
input_data = input_data.drop('Geography', axis=1)

# Label encode the 'Gender' column
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# Display the prediction
if st.button("Predict"):
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    st.write(f"Predicted Salary: {prediction[0][0]:.2f}")

