import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

#st.set_option('deprecation.showPyplotGlobalUse', False) # Eliminamos los warnings



st.markdown("<p style='font-size: 10px;'>Los datos utilizados para este proyecto, son oficial de la Union Europea. <a href='https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b'>Datos</a></p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; text-decoration: underline; color: skyblue;'>Emisiones de CO2 vehiculos</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

url = ''


#
st.markdown("<p style='font-size: 17px; color: whitebone;'>&nbsp;&nbsp;&nbsp; El <strong>calentamiento global</strong> es un tema mundial. Los gases de efecto invernadero es una de las causas, y segun la union europea, la principal.</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 17px; color: whitebone;'>&nbsp;&nbsp;&nbsp; El <strong>CO2</strong>, dióxido de carbono, es el gas que emiten los vehiculos que utilizan combustible fosil.</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 12px; color: whitebone;'>La UE quiere reducir estas emisiones. Estos son los objetivos. <a href='https://www.europarl.europa.eu/topics/es/article/20180920STO14027/reducir-las-emisiones-de-los-automoviles-nuevos-objetivos-de-co2'>Datos UE</a></p>", unsafe_allow_html=True)

st.markdown("<p style='font-size: 12px;'>No tendremos en cuenta los vehiculos electricos e hibridos</p>", unsafe_allow_html=True)
st.markdown("- <p style='font-size: 25px; color: orange;'>buenas</p>", unsafe_allow_html=True)
st.markdown("- <p style='font-size: 25px; color: orange;'>buenas</p>", unsafe_allow_html=True)
st.markdown("- <p style='font-size: 25px; color: orange;'>buenas</p>", unsafe_allow_html=True)
st.markdown('--- \n --- \n<br>', unsafe_allow_html=True)

# Data
df = pd.read_csv('df_fit.csv')
df.drop(columns=['ID'], inplace=True)

#####################################################################
#Información general

st.markdown("<p style='font-size: 25px; text-align: center;'>Informacion general de los datos</p>", unsafe_allow_html=True)
st.dataframe(df.describe())

#Distribucion de las variables
fig, ax = plt.subplot(figsize=(15, 10))
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
fig, ax = plt.subplot()
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


st.text(f'Mean Squared Error: {mse}')
st.text(f'R^2 Score: {r2}')

#####################################################################
#Interpretacion de los datos y visualizacion
fig, ax = plt.subplot()
ax.scatter(y_test, y_pred)
ax.xlabel('Valores reales')
ax.ylabel('Valores predichos')
ax.title('Comparacion entre valores reales y predichos')
st.pyplot(fig)

#####################################################################
#Residuos

residuos = y_test - y_pred

#Grafico residuos vs predichos
fig, ax = plt.subplot()
ax.scatter(y_pred, residuos)
ax.axhline(y=0, color='r', linestyle='--')
ax.xlabel('Valores predichos')
ax.ylabel('Residuos')
ax.title('Grafico de residuos')
st.pyplot(fig)

#Histogrma de residuos
fig, ax = plt.subplot()
ax.hist(residuos, bins=30)
ax.xlabel('Residuos')
ax.ylabel('Frecuencia')
ax.title('Histograma de residuos')
st.pyplot(fig)

