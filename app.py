import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Head

st.title('Segmentation client')
st.markdown("Cette application retourne les segments attribues aux clients de deux manieres : ")
st.markdown(" - Une prediction pour un client unique")
st.markdown(" - Un fichier client complete par une colonne contenant les segments")

# Left

title_pred = st.sidebar.title("Client unique")
st.sidebar.subheader("Selectionner les differents attributs du nouveau client et le modele retournera le segment auquel il appartient")

with st.sidebar.form("my_form"):
	gender = st.selectbox('Sexe', ('Homme', 'Femme'))
	gender_map = {'Homme' : 'Male', 'Femme' : 'Female'}
	gender = pd.Series(gender).map(gender_map)[0]

	ever_married = st.selectbox('A deja ete marrie', ('Oui', 'Non'))
	married_map = {'Oui' : 'Yes', 'Non' : 'No'}
	ever_married = pd.Series(ever_married).map(married_map)[0]

	age = round(st.number_input('Age', step=1))

	graduated = st.selectbox('Diplome', ('Oui', 'Non'))
	graduated_map = {'Oui' : 'Yes', 'Non' : 'No'}
	graduated = pd.Series(graduated).map(graduated_map)[0]

	profession = st.selectbox('Profession', ('Sante', 'Ingenieur', 'Avocat', 'Divertissement', 'Artiste', 'Executant', 'Docteur', 'Artisan', 'Marketing'))
	profession_map = {'Sante' : 'Healthcare', 'Ingenieur' : 'Engineer', 'Avocat' : 'Lawyer', 'Divertissement' : 'Entertainment', 'Artiste' : 'Artist', 'Executant' : 'Executive', 'Docteur' : 'Doctor', 'Artisan' : 'Homemaker', 'Marketing' : 'Marketing'}
	profession = pd.Series(profession).map(profession_map)[0]

	work_experience = round(st.number_input("Nombre d'annees d'experience professionelle", step=1))

	spending_score = st.selectbox('Depense moyenne', ('Basse', 'Moyenne', 'Elevee'))
	spending_map = {'Basse' : 'Low', 'Moyenne' : 'Average', 'Elevee' : 'High'}
	spending_score = pd.Series(spending_score).map(spending_map)[0]

	family_size = round(st.number_input('Taille de la famille', step=1))

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

	submitted = st.form_submit_button("Predire")

	if submitted:
		st.subheader('Ce client appartient au segment ' + str(pred))

# Middle

st.subheader('Fichier client')
st.markdown('Le fichier doit avoir la structure suivante :')

new_customer_1 = np.array(['Male', 'Yes', 60, 'Yes', 'Artist', 1, 'High', 5, 'Cat_4'])
new_customer_2 = np.array(['Female', 'No', 26, 'Yes', 'Healthcare', 3, 'Low', 1, 'Cat_6'])
df_col = ['Sexe', 'A deja ete marrie', 'Age', 'Diplome', 'Profession', "Nombre d'annees d'experience", 'Depense moyenne', 'Taille de la famille', 'Categorie']
df = pd.DataFrame([new_customer_1, new_customer_2], columns=df_col)
df_view = st.table(df)

st.markdown('Le programme retourne :')

df['Segment'] = [0, 1]
df_view = st.table(df)

customer_file = st.file_uploader('Importer un fichier client')

@st.cache
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

if customer_file is not None:
    
   customer_file = pd.read_csv(customer_file)
   df_to_predict = customer_file.copy()
   df_to_predict.iloc[:, 0] = encoder_gender.transform(df_to_predict.iloc[:, 0])
   df_to_predict.iloc[:, 1] = encoder_ever_married.transform(df_to_predict.iloc[:, 1])
   df_to_predict.iloc[:, 3] = encoder_graduated.transform(df_to_predict.iloc[:, 3])
   df_to_predict.iloc[:, 4] = encoder_profession.transform(df_to_predict.iloc[:, 4])
   df_to_predict.iloc[:, 6] = encoder_spending_score.transform(df_to_predict.iloc[:, 6])
   df_to_predict.iloc[:, 8] = encoder_var_1.transform(df_to_predict.iloc[:, 8])

   pred = model.predict(scaler.transform(df_to_predict))
   customer_file['Segment'] = pred
   df_csv = convert_df(customer_file)

   st.download_button('Fichier client complete', df_csv, file_name='fichier_client.csv')
   
test = pd.read_csv('test.csv')
test = convert_df(test)

"""
traduire en anglais 
inclure try except pour generer exception si le fichier na pas la bonne taille
"""

with st.expander('Note'):
	st.write("Un fichier est disponible pour pouvoir tester l'application : ")
	st.download_button('Fichier test', test, file_name='test.csv')

	st.write("Cette application est basee sur un modele que j'ai cree et consultable dans ce notebook : https://www.kaggle.com/richez/customer-segmentation")
