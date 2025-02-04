import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_csv('iris.csv')
    df['species'] = df['species'].map({'setosa':1,'versicolor':2,'virginica':3})
    X = df.drop('species',axis=1)
    y = df['species']
    return df,X,y

df,X,y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=15)

model = RandomForestClassifier()
model.fit(X_train,y_train)


st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider('Sepal length',float(df['sepal_length'].min()),float(df['sepal_length'].max()))
sepal_width = st.sidebar.slider('Sepal width',float(df['sepal_width'].min()),float(df['sepal_width'].max()))
petal_length = st.sidebar.slider('Petal length',float(df['petal_length'].min()),float(df['petal_length'].max()))
petal_width = st.sidebar.slider('Petal width',float(df['petal_width'].min()),float(df['petal_width'].max()))

# sepal_length = input('enter sepal length: ')
# sepal_width = input('enter sepal width: ')
# petal_length = input('enter petal length: ')
# petal_width = input('enter petal width: ')

input_data = [[sepal_length,sepal_width,petal_length,petal_width]]

y_pred = model.predict(input_data)
Species_map = {1:'setosa',2:'versicolor',3:'virignica'}
predicted_species = Species_map[y_pred[0]]

st.write('Prediction')
st.write(f'The Predicted Species is: {predicted_species}')