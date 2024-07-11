import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False) # Delete warning


st.markdown("<h1 style='text-align: center; text-decoration: underline; color: skyblue;'>Emisiones de CO2 vehiculos</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px;'>Los datos utilizados para este proyecto, son datos ooficial es de la Union Europea. <a href='https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b'>Datos</a></p>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("<p style='font-size: 30px;'>Cuanto contaminan los vehiculos? <br> Los vehiculos electricos no producen emisiones de CO2</p><br><br>", unsafe_allow_html=True)

df = pd.read_csv('df_fit.csv')

gr1 = df[['Mass_(kg)', 'CO2_emission_(g/km)', 'Engine_size', 'Fuel_consumption_(l/100km)']]

gr1.hist(figsize=(8, 8))
st.pyplot(plt.show())


plt.scatter(df['Fuel_consumption_(l/100km)'], df['CO2_emission_(g/km)'], color='blue')
plt.xlabel('Fuel consumption')
plt.ylabel('CO2 emission')
st.pyplot(plt.show())