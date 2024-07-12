#importation des librairies necessaires
import joblib
import numpy as np
import streamlit as st
import sklearn
import pickle as pkl

path="E:/exerces G-6/AED/exos_ML_deployements/inclusion_financier_bank/Financial_inclusion_bank/datasets_db"
# Presentation de l'application
st.title("Finding-CHURN_BANKER_ACCOUNT")
st.subheader("Réalisée par: DAOUDA BAMIS DIALLO")
st.markdown("Cette application à pour but d'aider les responsables des structures bancaires à trouver des nouveaux clients succeptibles d'utile ou créer un compte bancaire")

# chargement du modele
@st.cache_data(persist=True)
def load_models():
    file = open("model_FI_end.pkl", "rb")
    data= pkl.load(file)
    file.close()
    return data

modele_end = load_models()


# modele_end = joblib.load(filename="model_fi1_bank.joblib")

# la fonction d'interference
def inference(age_of_respondent,household_size,cellphone_access,no_formal_education,primary_education,
              location_type,country_Kenya,job_Formally_employed_Private,tertiary_education,gender,vocational_Specialised_training,econdary_ducation):
    
    df = np.array([age_of_respondent,household_size,cellphone_access,no_formal_education,primary_education,
              location_type,country_Kenya,job_Formally_employed_Private,tertiary_education,gender,vocational_Specialised_training,econdary_ducation])
    predict = modele_end.predict(df.reshape(1,-1))
    return predict

# saisie des informations 
st.write("Veillez entrer les informations de la personne")
                                  
age_of_respondent = st.number_input(label="Age", value=0.22619)
household_size  = st.number_input(label="Taille de l'entreprise", value=0.10)
cellphone_access= st.number_input(label="Telephone",value=1.0)
no_formal_education = st.number_input(label="Pas d'education formel?")
primary_education = st.number_input(label="Niveau d'education primaire")
location_type = st.number_input(label="Type de locat")
country_Kenya = st.number_input(label="Habitez-vous aau kenya?")
job_Formally_employed_Private = st.number_input(label="Emplyé privé d'un travail formel?")
tertiary_education = st.number_input(label="Niveau d'education tertiare?")
gender= st.number_input(label="Genrre")
vocational_Specialised_training = st.number_input(label=" Vocational/Specialised training?")
econdary_ducation= st.number_input(label="Niveau d'education secondaire?")

# creation du bouton de prediction
if st.button('Predire'):
    prediction = inference(age_of_respondent,household_size,cellphone_access,no_formal_education,primary_education,
              location_type,country_Kenya,job_Formally_employed_Private,tertiary_education,gender,vocational_Specialised_training,econdary_ducation)
    if prediction[0]==1:
        st.success("Suceptible d'utiliser un compte")
    elif prediction[0]==0:
        st.warning("Suceptible de pas utiliser un compte")