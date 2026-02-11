import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit.components.v1 as components

#st.set_option('deprecation.showPyplotGlobalUse', False) # Eliminamos los warnings



st.markdown("<p style='font-size: 10px;'>Los datos utilizados para este proyecto, son oficial de la Union Europea. <a href='https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b'>Datos</a></p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: skyblue;'>Emisiones de CO2 en vehiculos</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

#
st.markdown("<p style='font-size: 17px; color: whitebone;'>&nbsp;&nbsp;&nbsp; El <strong>calentamiento global</strong> es un tema mundial. Los gases de efecto invernadero es una de las causas, y segun la union europea, la principal.</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 17px; color: whitebone;'>&nbsp;&nbsp;&nbsp; El <strong>CO2</strong>, dióxido de carbono, es el gas que emiten los vehiculos que utilizan combustible fosil y donde la UE hace hincapié en reducir estas emisiones.</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 12px; color: whitebone;'>Estos son los objetivos. <a href='https://www.europarl.europa.eu/topics/es/article/20180920STO14027/reducir-las-emisiones-de-los-automoviles-nuevos-objetivos-de-co2'>Datos UE</a></p>", unsafe_allow_html=True)

with st.expander("EU objetives 01/02/2026"): 
    Col1, Col2, Col3 = st.columns([1, 2, 1])
    with Col2: 
        st.image("images/EU_objectives_01022026.png", caption="Objetivos de la UE para reducir las emisiones de CO2 en vehiculos", use_container_width=True)

page1_url = 'pages/clean_data.html'
with st.expander("Como se limpiaron los datos", expanded=True):
    with open(page1_url,  'r') as file:
        html_content = file.read()
    components.html(html_content, height=600, scrolling=True)

#st.markdown("- <p style='font-size: 25px; color: orange;'>buenas</p>", unsafe_allow_html=True)
#st.markdown("- <p style='font-size: 25px; color: orange;'>buenas</p>", unsafe_allow_html=True)
#st.markdown("- <p style='font-size: 25px; color: orange;'>buenas</p>", unsafe_allow_html=True)
st.markdown('--- \n --- \n<br>', unsafe_allow_html=True)

# Data
df = pd.read_csv('df_fit.csv')
df.drop(columns=['ID'], inplace=True)

#####################################################################
#Información general

st.markdown("<p style='font-size: 25px; text-align: center;'>Informacion general de los datos</p>", unsafe_allow_html=True)
st.dataframe(df.describe())

#Distribucion de las variables
fig, ax = plt.subplots(figsize=(15, 10))
df.hist(bins=30, ax=ax)
plt.tight_layout()
st.pyplot(fig)
st.markdown('---')

#Relacion entre las variables
container = st.container()

def generate_pairplot():
    fig = plt.figure()
    sns.pairplot(df)
    st.pyplot(fig)
st.markdown("<p style='font-size: 25px;'>Relacion entre las variables.</p> Este grafico requiere de 2 min para mostrarse", unsafe_allow_html=True)
if st.button('Gráfico'):
    generate_pairplot()
st.markdown('--- \n <br>', unsafe_allow_html=True)

#####################################################################
#Análisis

df_fit = df
df_fit = pd.get_dummies(df_fit, columns=['Fuel_type'])
df_fit.rename(columns={'Fuel_type_petrol': 'petrol', 'Fuel_type_diesel': 'diesel'})


st.markdown("<p style='font-size: 25px; text-align: center;'>Mapa de calor</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 25px;'>Correlacion de Pearson</p><br> Relacion lineal entre dos variables", unsafe_allow_html=True)
st.markdown("- 1: Correlacion positiva perfecta. Cuando una variable sube, la otra sube", unsafe_allow_html=True)
st.markdown("- -1: Correlacion negativa perfecta. Cuando una variable sube, la otra baja", unsafe_allow_html=True)
st.markdown("- 0: No hay correlacion. No hay alguna correlacion entre las variables", unsafe_allow_html=True)

corr_matrix = df_fit[['Mass_(kg)', 'Engine_size', 'Fuel_consumption_(l/100km)', 'CO2_emission_(g/km)']].corr() #Relacion entre 2 variables. 1 = Ambas variables aumentan en una proporcion fija. -1 = Una crece y al otra crece en una proporcion fija. 0 = Ninguna correlacion 
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
st.pyplot(fig)
st.write(corr_matrix)

st.markdown("&nbsp;&nbsp;&nbsp; Bajar el consumo de combustible es lo primordial para bajar las emisiones de CO2. La relacion **consumo-emisiones** es cercana a perfecta positiva", unsafe_allow_html=True)

#####################################################################
#Preprocesamiento y modelo

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['Mass_(kg)', 'Engine_size', 'Fuel_consumption_(l/100km)', 'CO2_emission_(g/km)']
df_scaled = df_fit[numerical_cols]
df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=numerical_cols) #fit = Calcula la media y la desviacion estandar. transform = Transforma para que tenga una media de 0 y una desviacion estandar de 1

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
mse = mean_squared_error(y_test, y_pred) #Media de los cuadrados de los errores. Diferencia promedio al cuadrado entre los valores predichos y los reales.
r2 = r2_score(y_test, y_pred) #Proporcion de la variacion en la variable dependiente. Que tan bien los datos de entrenamiento se ajustan al modelo

st.text(f'El modelo es tan preciso que la diferencia de error entre lo que se predice y lo real es menos a 0.1. Es decir, que se puede confiar ampliamente en las predicciones. \n Mean Squared Error: {mse}')
st.text(f'R^2 Score: {r2}')

#####################################################################
#Interpretacion de los datos y visualizacion
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('Valores reales')
ax.set_ylabel('Valores predichos')
ax.set_title('Comparacion entre valores reales y predichos')
st.pyplot(fig)

#####################################################################
#Residuos

residuos = y_test - y_pred

#Grafico residuos vs predichos
fig, ax = plt.subplots()
ax.scatter(y_pred, residuos)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Valores predichos')
ax.set_ylabel('Residuos')
ax.set_title('Grafico de residuos')
st.pyplot(fig)

#Histogrma de residuos
fig, ax = plt.subplots()
ax.hist(residuos, bins=30)
ax.set_xlabel('Residuos')
ax.set_ylabel('Frecuencia')
ax.set_title('Histograma de residuos')
st.pyplot(fig)

