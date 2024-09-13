
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Charger le modèle pré-entraîné
model = joblib.load('random_forest_model.pkl')

# Fonction pour prédire les prix
def predict_price(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

# Interface utilisateur
st.title('Prédiction des Prix des Ordinateurs')

st.sidebar.header('Entrez les caractéristiques de l'ordinateur')

# Exemple de caractéristiques à entrer
inches = st.sidebar.number_input('Taille de l'écran (en pouces)', min_value=10, max_value=20, value=15)
ram = st.sidebar.selectbox('RAM (en Go)', [4, 8, 16, 32])
# Ajoutez plus de champs selon les caractéristiques de votre modèle

features = [inches, ram]  # Ajoutez plus de caractéristiques ici

if st.sidebar.button('Prédire'):
    price = predict_price(features)
    st.write(f'Le prix estimé est : {price:.2f} €')
