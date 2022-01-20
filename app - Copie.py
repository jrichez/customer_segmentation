import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Segmentation client')
st.header("Cette application permet d'attribuer un nouveau client a un cluster")
st.subheader("Selectionner les differents attributs du nouveau client et le modele retournera le segment auquel il appartient")

gender = st.selectbox('Sexe', ('Homme', 'Femme'))
gender_map = {'Homme' : 'Male', 'Femme' : 'Female'}
gender = pd.Series(gender).map(gender_map)[0]

ever_married = st.selectbox('A deja ete marrie', ('Oui', 'Non'))
married_map = {'Oui' : 'Yes', 'Non' : 'No'}
ever_married = pd.Series(ever_married).map(married_map)[0]

age = round(st.number_input('Age'))

graduated = st.selectbox('Diplome', ('Oui', 'Non'))
graduated_map = {'Oui' : 'Yes', 'Non' : 'No'}
graduated = pd.Series(graduated).map(graduated_map)[0]


profession = st.selectbox('Profession', ('Sante', 'Ingenieur', 'Avocat', 'Divertissement', 'Artiste', 'Executant', 'Docteur', 'Artisan', 'Marketing'))
profession_map = {'Sante' : 'Healthcare', 'Ingenieur' : 'Engineer', 'Avocat' : 'Lawyer', 'Divertissement' : 'Entertainment', 'Artiste' : 'Artist', 'Executant' : 'Executive', 'Docteur' : 'Doctor', 'Artisan' : 'Homemaker', 'Marketing' : 'Marketing'}
profession = pd.Series(profession).map(profession_map)[0]

work_experience = round(st.number_input("Nombre d'annees d'experience professionelle"))

spending_score = st.selectbox('Depense moyenne', ('Basse', 'Moyenne', 'Elevee'))
spending_map = {'Basse' : 'Low', 'Moyenne' : 'Average', 'Elevee' : 'High'}
spending_score = pd.Series(spending_score).map(spending_map)[0]

family_size = round(st.number_input('Taille de la famille'))

var_1 = st.selectbox('Categorie', ('1', '2', '3', '4', '5', '6', '7'))
var_1_map = {'1' : 'Cat_1', '2' : 'Cat_2', '3' : 'Cat_3', '4' : 'Cat_4', '5' : 'Cat_5', '6' : 'Cat_6', '7' : 'Cat_7'}
var_1 = pd.Series(var_1).map(var_1_map)[0]

new_customer = np.array([gender, ever_married, age, graduated, profession, work_experience, spending_score, family_size, var_1])


encoder_gender = pickle.load(open('encoder_gender.pkl', 'rb'))
encoder_ever_married = pickle.load(open('encoder_ever_married.pkl', 'rb'))
encoder_graduated = pickle.load(open('encoder_graduated.pkl', 'rb'))
encoder_profession = pickle.load(open('encoder_profession.pkl', 'rb'))
encoder_spending_score = pickle.load(open('encoder_spending_score.pkl', 'rb'))
encoder_var_1 = pickle.load(open('encoder_var_1.pkl', 'rb'))

new_customer[0] = encoder_gender.transform(new_customer[0].reshape(1, -1))[0]
new_customer[1] = encoder_ever_married.transform(new_customer[1].reshape(1, -1))[0]
new_customer[3] = encoder_graduated.transform(new_customer[3].reshape(1, -1))[0]
new_customer[4] = encoder_profession.transform(new_customer[4].reshape(1, -1))[0]
new_customer[6] = encoder_spending_score.transform(new_customer[6].reshape(1, -1))[0]
new_customer[8] = encoder_var_1.transform(new_customer[8].reshape(1, -1))[0]

scaler = pickle.load(open('scaler.pkl', 'rb'))

new_customer = scaler.transform(new_customer.reshape(1, -1))

model = pickle.load(open('customer_segmentation.sav', 'rb'))
pred = model.predict(new_customer)

click = st.button('Predire')

if click:
	st.subheader('Ce client est attribue au segment ' + str(pred))


with st.expander('Note'):
	st.write("Cette application est basee sur un modele que j'ai cree et consultable dans ce notebook : https://www.kaggle.com/richez/customer-segmentation")
