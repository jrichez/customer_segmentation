import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Uber pickups in NYC')

filename = 'customer_segmentation.sav'
loaded_model = pickle.load(open(filename, 'rb'))

Gender : selectbox
Ever_Married : selectbox
Age : number_input 
Graduated : selectbox 
Profession : selectbox 
Work_Experience : number_input 
Spending_Score : selectbox 
Family_Size : number_input 
Var_1 : selectbox