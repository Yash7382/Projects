import streamlit as st
import pickle
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

model = keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction')

## user data
geo = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('No. of Products',1,4)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],   
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

## one hot encode geo
geo_encoded = onehot_encoder_geo.transform([[geo]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## combine one hot encoded data and geo encoded data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis = 1)

## scale the input data
input_data_scaled = scaler.transform(input_data)

## prediction churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write('The person is likely to churn')
else:
    st.write('The person is not likely to churn')
