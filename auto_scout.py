import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sklearn as sklearn
import pickle

#image = Image.open("img.jpg")
#st.image(image,width=100)
st.title('Car Price Prediction')

mlmodel = st.selectbox('Please select the maschine learning model',['Lasso','KNN','RandomForest','XGBoost'])


brand = st.radio("Select a brand", ("Audi", "Renault", "Opel"))
if brand == 'Audi':
    model = st.radio('Select a model:',('A1','A3'))
elif brand == 'Renault':
    model = st.radio('Select a model',('Clio','Duster','Espace'))
elif brand == 'Opel':
    model = st.radio('Select a model',('Astra','Corsa','Insignia'))

km = st.slider('Select km',0,500000,1000,1000)

age = st.slider('Select age',1,4,1,1)

gear = st.radio('Select gearing type',('Automatic','Manual','Semi-automatic'))

kw = st.slider('Select kw',40,300,50,1)
my_dict = {
    "hp_kw": kw,
    "age": age,
    "km": km,
    "make_model": brand + '_' + model,
    "gearing_type": gear
}

my_dict = pd.DataFrame([my_dict])


columns=[
 'hp_kw',
 'km',
 'age',
 'make_model_Audi_A1',
 'make_model_Audi_A3',
 'make_model_Opel_Astra',
 'make_model_Opel_Corsa',
 'make_model_Opel_Insignia',
 'make_model_Renault_Clio',
 'make_model_Renault_Duster',
 'make_model_Renault_Espace',
 'gearing_type_Automatic',
 'gearing_type_Manual',
 'gearing_type_Semi-automatic']


my_dict_dummy = pd.get_dummies(my_dict).reindex(columns=columns, fill_value=0)

final_scaler = pickle.load(open('scaler_reg.pkl', "rb"))
my_dict_scaled = final_scaler.transform(my_dict_dummy)


filename = "reg_final.pkl"
model = pickle.load(open(filename, "rb"))
pred = model.predict(my_dict_scaled)
x=pred[0]
x=x.astype(int)
st.success(x)










