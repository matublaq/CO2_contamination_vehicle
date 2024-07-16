import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False) # Delete warning


st.markdown("<p style='font-size: 10px;'>Los datos utilizados para este proyecto, son datos ooficial es de la Union Europea. <a href='https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b'>Datos</a></p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; text-decoration: underline; color: skyblue;'>Emisiones de CO2 vehiculos</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("- <p style='font-size: 20px; color: orange;'>Que factores influyen en la emision de CO2?</p>", unsafe_allow_html=True)
st.markdown("- <p style='font-size: 20px; color: orange;'>Que factores influyen mas en el consumo de combustible?</p>", unsafe_allow_html=True)
st.markdown("- <p style='font-size: 20px; color: orange;'>Se puede predecir el consumo de combustibles o las emisiones de CO2 basandose en otras caracteristicas del vehiculo?</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 12px;'>No tendremos en cuenta los vehiculos electricos o hibridos</p><br><br>", unsafe_allow_html=True)


# Data
df = pd.read_csv('df_fit.csv')
df.drop(columns=['ID'], inplace=True)

#####################################################################
#Informacion general de los datos

st.dataframe(df.describe())

#Distribucion de las variables
df.hist(bins=30, figsize=(15, 10))
st.pyplot(plt.show())

#Relacion entre las variables
sns.pairplot(df)
st.pyplot(plt.show())