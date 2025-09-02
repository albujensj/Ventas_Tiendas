import pandas as pd
import numpy as np

# Cargar dataset
df = pd.read_csv(r"c:\Users\Sarita\OneDrive\Escritorio\python Machine Learning - Trimestre 6\Ejercicio regresión lineal\regresion_lineal_ventas\ventas_tiendas.csv")


# Vista previa
print("Primeras filas:")
print(df.head())

# Info general
print("\nInformación general:")
print(df.info())

# Estadísticas
print("\nEstadísticas descriptivas:")
print(df.describe(include="all"))


# 3.3 Eliminar filas donde falten datos
df = df.dropna(how="any")

# 3.5 Eliminar outliers (ventas negativas o exageradas)
df = df[(df["ventas"] > 0)]

#Convertir la ubicacion en valores numericos
df["ubicacion"] = df["ubicacion"].replace({
    "suburbana": 1,
    "urbana": 2,
    "rural": 3
})

# 3.4 Eliminar duplicados
df_new = df.drop_duplicates()
df_new.info()
print(df_new.head(10))


#Modelado

# crear csv
df_new.to_csv('ventas_tiendas_limpio')
