import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False) # Delete warning


st.markdown("<p style='font-size: 10px;'>Los datos utilizados para este proyecto, son datos ooficial es de la Union Europea. <a href='https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b'>Datos</a></p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; text-decoration: underline; color: skyblue;'>Emisiones de CO2 vehiculos</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<p style='font-size: 30px;'>Cuanto contaminan los vehiculos?</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 12px;'>No tendremos en cuenta los vehiculos electricos o hibridos</p><br><br>", unsafe_allow_html=True)

# Data
df = pd.read_csv('df_fit.csv')

#Cuales son las medias de peso, emisiones, tamano de motor y consumo de combustible
gr1 = df[['Mass_(kg)', 'CO2_emission_(g/km)', 'Engine_size', 'Fuel_consumption_(l/100km)']]
gr1.hist(figsize=(8, 8))
st.pyplot(plt.show())

#Como se relaciona el consumo del combustible con las emisiones
plt.scatter(df['Fuel_consumption_(l/100km)'], df['CO2_emission_(g/km)'], color='blue')
plt.xlabel('Fuel consumption')
plt.ylabel('CO2 emission')
st.pyplot(plt.show())

#Como se relaciona el tamano del motor con las emisiones
plt.scatter(df['Engine_size'], df['CO2_emission_(g/km)'], color='blue')
plt.xlabel('Engine size')
plt.ylabel('CO2 emission')
st.pyplot(plt.show())


# Modelo de Regresion lineal
st.markdown("<br><br><p style='font-size: 20px; text-align: center; color: orange; text-decoration: underline'>Modelo de regresion lineal</p>", unsafe_allow_html=True)

msk = np.random.rand(len(df)) < 0.8 #msk = mask. Lista de len(df) numeros aleatorios entre el 0 y 1. aproximadamente el 80% del conjunto (<0.8)
train = df[msk]
test = df[~msk]

from sklearn import linear_model

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Engine_size']])
train_y = np.asanyarray(train[['CO2_emission_(g/km)']])
regr.fit(train_x, train_y)

#Coeficientes
st.text(f'Pendiente: {round(regr.coef_[0][0], 2)}')
st.text(f'Intersección: {round(regr.intercept_[0], 2)}')

#Grafica del resultado
plt.scatter(train['Engine_size'], train['CO2_emission_(g/km)'], color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('Engine size')
plt.ylabel('CO2 emission')
st.pyplot(plt.show())

#Testeo
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Engine_size']])
test_y = np.asanyarray(test[['CO2_emission_(g/km)']])
test_y_ = regr.predict(test_x)

st.text(f'Error medio absoluto: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
st.text(f'Resudial suma de los cuadrados (MSE): %.2f' % np.mean((test_y_ - test_y)**2))
st.text(f'R2-score: %.2f' % r2_score(test_y, test_y_))