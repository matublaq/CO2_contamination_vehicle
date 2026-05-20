#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import shutil

import requests

import time
import numpy as np

import pandas as pd
import polars as pl
import polars.selectors as cs


import duckdb

import pyarrow.parquet as pq


# In[8]:


os.environ["POLARS_TEMP_DIR"] = "D:\\TEMP_DIR"
os.environ["TMPDIR"] = "D:\\TEMP_DIR"
os.environ["TEMP"] = "D:\\TEMP_DIR"
os.environ["TMP"] = "D:\\TEMP_DIR"

duckdb.sql("""
    SET temp_directory = "D:\\TEMP_DIR"
""")


# In[9]:


import os

# Verificación de la variable de entorno
temp_path = os.environ.get("POLARS_TEMP_DIR")

if temp_path:
    print(f"✅ Polars usará: {temp_path}")
    if not os.path.exists(temp_path):
        print("⚠️ Advertencia: La carpeta no existe, Polars podría fallar o usar el default.")
else:
    print("❌ La variable no está seteada. Polars usará el disco C (Temp del sistema).")


# ## <p style='text-align: center; text-decoration: underline; color: #10A0B4;'> Datos oficiales de la **Union Europea** </p>
# [Origen de los datos](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b)

# ---

# ## Autos registrados desde **2010** hasta **2023**

# ---

# #### El dataset contiene mas de **16 Gigas** de informacion (80.378.486 filas).
# 
# El gran tamano del conjunto de datos trae problemas:
# - No puedo subirlo a GitHub (+100MB). 
# - Lento procesamiento.
# 
# Vamos a resolver ambas cuentiones achicando el dataset de manera tal que el impacto por usar menos informacion sea minimo

# ---

# El archivo pesa +16GB y tiene más de 80 millones de filas.

# In[10]:


url = 'ignore_data.csv'
file_weight = os.path.getsize(url)
print('Database size: ', round((file_weight/1024)/1024/1024, 2), 'GB')

count = duckdb.sql("SELECT count(*) FROM ignore_data.csv").fetchone()[0]
print(f"El archivo .csv tiene {count} registros.")

print(pl.scan_csv('ignore_data.csv').collect_schema().names()) # 10 seg aprox.
# duckdb.sql("SELECT * FROM 'ignore_data.csv' LIMIT 0").show() # 10 seg aprox. 
# duckdb.sql("SUMMARIZE ignore_data.csv").show(max_rows=100) # 10 seg aprox. 
# pl.scan_parquet("ignore_data_complete_clean.parquet").describe().head(45) # 45 seg aprox.

# In[11]:


columns_to_load = ['ID', 'year', 'Country', 'Mh', 'z (Wh/km)', 'ep (KW)', 'Mk', 'Cn', 
                   'Electric range (km)', 'm (kg)', 'Ewltp (g/km)', 'Ft', 'ec (cm3)', 
                   'Fuel consumption ', 'MMS']

rename_columns = {
    'm (kg)': 'mass_kg', 'Ewltp (g/km)': 'co2_emission_g/km', 'Ft': 'fuel_type',
    'ec (cm3)': 'engine_size_cm3', 'Fuel consumption ': 'fuel_consumption_l/100km',
    'year': 'year_of_fabrication', 'Mh': 'manufacturer', 'z (Wh/km)': 'energy_consumption_Wh/km',
    'ep (KW)': 'electric_power_KW', 'Mk': 'model', 'Cn': 'commercial_name',
    'Electric range (km)': 'electric_range_km', 'MMS': 'manufacturar_name', "Country": "country"
}

# scan_csv (Lazy) - Esto NO carga el archivo en RAM
df_lazy = (
    pl.scan_csv('ignore_data.csv', infer_schema_length=10000) # En base a las primeras 10k filas, Polars infiere el tipo de cada columna.
    .select(columns_to_load)
    .rename({k: v for k, v in rename_columns.items() if k in columns_to_load})
    .drop_nulls(subset=['model', 'commercial_name', 'fuel_type', 'year_of_fabrication'])
    .with_columns(cs.string().str.strip_chars().str.to_lowercase().replace("", None))
)

print("Iniciando procesamiento en streaming...")
df_lazy.sink_parquet(f"ignore_data_complete_clean.parquet") # 2 min aprox. para procesar. 
print("Done!")

file_weight = os.path.getsize("ignore_data_complete_clean.parquet")
print('.parquet size: ', round((file_weight/1024)/1024/1024, 2), 'GB')


# #### Unificamos datos. 

# In[12]:


duckdb.sql("""
    SELECT fuel_type, count(*) as total_fueltype
    FROM 'ignore_data_complete_clean.parquet'
    GROUP BY fuel_type
    ORDER BY total_fueltype DESC
""")


# In[13]:


mapeo_manufacturers = {
    "": ["aa-iva", "duplicate", "out of scope", "cng technik", "unknown", "aa-nss", "zhaoqing"], 
    "bmw": ["bmw ag", "bmw gmbh", "bayerische motoren werke ag", "bmw m gmbh"], 
    "volkswagen": [], 
    "toyota": ["toyota motor corporation", "toyota motor europe"], 
    "ford": ["ford werke gmbh", "ford india", "ford motor company", "ford-werke gmbh"], 
    "audi": ["audi ag", "audi sport", "audi hungaria", "quattro"],
    "peugeot": ["automobiles peugeot"],
    "citroen": ["automobiles citroen"],
    "fiat": ["fiat group", "fiat group automobiles spa"], 
    "mercedes-benz": ["mercedes-benz ag", "daimler ag", "mercedes amg", "mercedes-amg"],
    "opel": ["opel automobile"],
    "stellantis": ["psa", "stellantis europe"], 
    "hyundai": ["hyundai czech", "hyundai assan", "hyundai europe"],
    "kia": ["kia slovakia"],
    "jaguar": ["jaguar land rover limited", "jaguar cars ltd"],
    "suzuki": ["suzuki motor corporation", "magyar suzuki", "suzuki thailand", "maruti suzuki", "magyar suzuki corporation ltd"],
    "nissan": ["nissan automotive europe", "nissan international sa"],
    "honda": ["honda motor co", "honda uk", "honda turkiye", "honda china", "honda of the uk manufacturing", "honda automobile china co", "honda turkiye as", "honda automobile thailand co", "honda thailand"], 
    "mitsubishi": ["mitsubishi motors corporation", "mitsubishi motors thailand", "mitsubishi motors europe", "mitsubishi motors corporation mmc"], 
    "saic": ["saic motor corporation", "saic motor"], 
    "dr": ["dr automobiles", "dr motor"],
    "mazda": ["mazda europe", "mazda motor corporation"], 
    "kgm": ["ssangyong", "kg mobility"], 
    "mg": ["mg motor"], 
    "gm": ["gm korea", "gm korea company", "gm daewoo auto u tech comp"], 
    "subaru": ["fuji heavy industries", "fuji heavy industries ltd"],
    "general motors": ["general motors holdings", "general motors company", "gm italia"], 
    "mahindra & mahindra": ["mahindra"], 
    "chrysler": ["chrysler group llc"],
    "jmc": ["jiangling motor", "jiangxi jiangling", "jiangling motors"], 
    "faw": ["china faw"], 
    "saab": ["saab automobile ab"],
    "rolls-royce": ["rolls royce", "rolls-royce motor cars ltd"], 
    "aston Martin": ["aston martin lagonda ltd"], 
    "bluecar": ["bluecar italy", "vehicules electriques pininfarina-bollore s.a.s.", "vepb"],
    "bugatti": ["bugatti rimac"], 
    "chevrolet": ["chevrolet italia"],
    "daihatsu": ["daihatsu motor co"], 
    "dongfeng": ["dongfeng motor", "dongfeng liuzhou", "dongfeng motor corporation"], 
    "e-go": ["next ego mobile"], 
    "iveco": ["iveco spa"], 
    "lanzhou": ["lanzhou zhidou"], 
    "levc": ["lti carbodies"], 
    "lotus": ["wuhan lotus", "lotus group plc"], 
    "micro-vett": ["micro vett"],
    "osv": ["osv opel special vehicles"], 
    "radical motorsport": ["radical motorsport ltd", "radical motosport"], 
    "renault": ["renault trucks", "sovab"], 
    "zotye": ["zotye holding ng group"]
}
dic_aux_manufacturers = {}
for correcto, lista_errores in mapeo_manufacturers.items():
    for error in lista_errores: 
        dic_aux_manufacturers[error] = correcto
#print(dic_aux_manufacturers)
#print(len(dic_aux_manufacturers))

mapeo_fueltype = { 
    "hybrid_petrol": ["petrol-electric", "petrol/electric", "hybrid/petrol/e", "petrol phev"], 
    "hybrid_diesel": ["diesel/electric", "diesel-electric"], 
    "biofuel": ["e85", "biodiesel", "ng-biomethane", "ng_biomethane"], 
    "natural_gas": ["ng", "cng", "gnl", "lpg", "petrol-gas"], 
    "": ["unknown", "other", "NULL"]

}
dic_aux_fueltype = {}
for correcto, lista_errores in mapeo_fueltype.items():
    for error in lista_errores: 
        dic_aux_fueltype[error] = correcto
#print(dic_aux_fueltype)
#print(len(dic_aux_fueltype))


input_path = "ignore_data_complete_clean.parquet"
temp_path = "temporal_ignore_data_complete_clean.parquet"

query = (
    pl.scan_parquet(input_path)
    .with_columns([
        (pl.col("engine_size_cm3") / 1000).round(2).alias("engine_size_L"),

        pl.col("manufacturer")
        .str.strip_chars() # Elimina espacios en blanco al inicio y al final de la cadena.
        .str.to_lowercase() # Convierte la cadena a minúsculas para estandarizarla.
        .replace_strict(dic_aux_manufacturers, default=pl.col("manufacturer")), # Reemplaza los valores según el diccionario, si no encuentra una coincidencia, deja el valor original.

        pl.col("fuel_type")
        .str.strip_chars()
        .str.to_lowercase()
        .replace_strict(dic_aux_fueltype, default=pl.col("fuel_type"))
    ]).drop("engine_size_cm3") # Eliminamos la columna original para que no esté duplicada

    .with_columns(
        pl.col(["fuel_consumption_l/100km", "engine_size_L"]).cast(pl.Float32), 
        pl.col("mass_kg").cast(pl.Int16)
    )
)
query.sink_parquet(temp_path, compression="zstd", compression_level=15)
shutil.copy2(temp_path, "ignore_backup_1.parquet")
os.remove(input_path)
os.rename(temp_path, input_path)

file_weight = os.path.getsize(input_path)
print('.parquet size: ', round((file_weight/1024)/1024/1024, 2), 'GB')


# #### El auto que menos consume en el mundo: 
#     - Diesel es el Renault Clio Blue dCi 100 con un consumo de 3,6l/100km. 
#     - Petrol, es el Suzuki Swift 1.2 con 4,4l/100km.  
# #### El auto más pesado y liviano del mundo: 
#     - Más pesado es el Rolls-Royce Specrte con 2975kg
#     - Más liviano es el suzuki alto (versíon Japonesa) con 810kg

# In[ ]:


count = duckdb.sql("SELECT COUNT(*) FROM 'ignore_data_complete_clean.parquet'").fetchone()
coun_delete = duckdb.sql("""
    SELECT COUNT(CASE WHEN ("fuel_consumption_l/100km" < 3.6 AND "fuel_type" IN ('petrol', 'diesel')) OR ("mass_kg" <= 800 OR "mass_kg" >= 3000) THEN 1 END) AS eliminar
    FROM 'ignore_data_complete_clean.parquet'
""").fetchone()
print(f"Cantidad de registros antes de la limpieza: {count}")
print(f"Cantidad de registros eliminados: {coun_delete}")

######## Si tengo más de 32GB de RAM, puedo cargar el .parquet en memoria y hacer las transformaciones sin necesidad de escribir en disco. 
# Eliminamos los registros que no tienen sentido. 
# df = pl.read_parquet("ignore_data_complete_clean.parquet")
# df = df.filter(
#     (
#         # Condition to remove. 
#         (pl.col("fuel_consumption_l/100km") < 3.6) & (pl.col("fuel_type").is_in(["petrol", "diesel"])) 
#         |
#         (pl.col("mass_kg") <= 800) | (pl.col("mass_kg") >= 3000)
#     ).not_().fill_null(True)
# )   
# duckdb.sql("SELECT COUNT(*) FROM df ").show()
# df.write_parquet("ignore_data_complete_clean.parquet")
# duckdb.sql("SELECT COUNT(*) FROM 'ignore_data_complete_clean.parquet'").show()

input_path = "ignore_data_complete_clean.parquet"
temp_path = "temporal_ignore_data_complete_clean.parquet"

query = (
    pl.scan_parquet(input_path)
    .filter(
        # Condition to remove. 
        (
            ((pl.col("fuel_consumption_l/100km") < 3.6) & (pl.col("fuel_type").is_in(["petrol", "diesel"]))) 
            | 
            ((pl.col("mass_kg") <= 800) | (pl.col("mass_kg") >= 3200))
        ).eq(False).fill_null(True)
    )
)

df_result_count = query.select(pl.len()).collect()
print("Cantidad de registros después de la limpieza: ", df_result_count[0, 0])
query.sink_parquet(temp_path)

os.remove(input_path) 
os.rename(temp_path, input_path)

#duckdb.sql("SELECT COUNT(*) FROM query").show()
count2 = duckdb.sql(f"SELECT COUNT(*) FROM '{input_path}'").fetchone()
print(f"Cantidad de registros después de la limpieza: {count2}")

file_weight = os.path.getsize(input_path)
print('.parquet size: ', round((file_weight/1024)/1024/1024, 2), 'GB')




input_path = "ignore_data_complete_clean.parquet"
temp_path = "temporal_ignore_data_complete_clean.parquet"

query = (
    pl.scan_parquet(input_path)
    .with_columns([
        pl.col("ID").cast(pl.Int32), 
        pl.col("year_of_fabrication").cast(pl.String),
        pl.col(["energy_consumption_Wh/km", "electric_power_KW", "electric_range_km", "mass_kg", "co2_emission_g/km"]).cast(pl.Int16), 
        pl.col(["fuel_consumption_l/100km", "engine_size_L"]).cast(pl.Float32), 
        pl.col(["country", "manufacturer", "model", "commercial_name", "fuel_type", "manufacturar_name"]).cast(pl.Categorical) # Categorical: Es un tipo de dato que almacena los valores únicos de una columna y luego asigna un código numérico a cada valor. Esto reduce el espacio de almacenamiento y mejora el rendimiento en operaciones de filtrado y agrupamiento.
    ])
)

# Sirve para que no me cargue en memoria el DataFrame. Pero ya que el .parquet es mas liviano, lo puedo cargar en memoria y hacer las transformaciones. En memoria es más rápido que escribir en disco. 
query.sink_parquet(temp_path, compression="zstd", compression_level=15) 
shutil.copy2(temp_path, "ignore_backup_1.parquet")

os.remove(input_path) 
os.rename(temp_path, input_path)

file_weight = os.path.getsize(input_path)
print('.parquet size: ', round((file_weight/1024)/1024, 2), 'MB')
print("Done!")


# #### Reescribimos bien los datos necesarios

# In[16]:


duckdb.sql("SELECT * FROM 'ignore_data_complete_clean.parquet' LIMIT 0").show()


# ---

# In[17]:


duckdb.sql("SUMMARIZE SELECT * FROM 'ignore_data_complete_clean.parquet'").show(max_rows=25)


# #### Convertir el archivo en html
# <code>jupyter nbconvert --to html clean_data.ipynb</code>
