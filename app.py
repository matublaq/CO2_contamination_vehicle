import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False) # Eliminamos los warnings


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
#Información general

st.dataframe(df.describe())

#Distribucion de las variables
df.hist(bins=30, figsize=(15, 10))
st.pyplot(plt.show())

#Relacion entre las variables
#sns.pairplot(df)
#st.pyplot(plt.show())

#####################################################################
#Preprocesamiento
df_fit = df
df_fit = pd.get_dummies(df_fit, columns=['Fuel_type'])
df_fit.rename(columns={'Fuel_type_petrol': 'petrol', 'Fuel_type_diesel': 'diesel'})
st.dataframe(df_fit)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['Mass_(kg)', 'Engine_size', 'Fuel_consumption_(l/100km)']
df_fit[numerical_cols] = scaler.fit_transform(df_fit[numerical_cols])
st.dataframe(df_fit)

#####################################################################
#Análisis y modelado

corr_matrix = df_fit.corr() #Relacion entre 2 variables. 1 = Ambas variables aumentan en una proporcion fija. -1 = Una crece y al otra crece en una proporcion fija. 0 = Ninguna correlacion 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt.show())
st.write(corr_matrix)

#####################################################################
#Modelo predictivo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = df_fit.drop(columns=['Fuel_consumption_(l/100km)'])
y = df_fit['Fuel_consumption_(l/100km)']

#Train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Linear regression model training
model = LinearRegression()
model.fit(x_train, y_train)

#Prediction
y_pred = model.predict(x_test)

#Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.text(f'Mean Squared Error: {mse}')
st.text(f'R^2 Score: {r2}')
