import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv(r"c:\Users\Sarita\OneDrive\Escritorio\python Machine Learning - Trimestre 6\Ejercicio regresión lineal\regresion_lineal_ventas\ventas_tiendas_limpio.csv")

x = df[['empleados', 'publicidad', 'ubicacion']]
y = df['ventas']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
print(X_train.head())
print(Y_train.head())

modelo = LinearRegression()
modelo.fit(X_train, Y_train)

Y_pred = modelo.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("MSE:", mse)
print("R2:", r2)

#Se muestra la ecuación 
print("Intercepto:", modelo.intercept_)
print("Coeficientes", modelo.coef_)
print(f"Ecuación: ventas={modelo.intercept_:.2f}, + {modelo.coef_[0]:.2f}, * empleados + {modelo.coef_[1]:.2f}, * publicidad + {modelo.coef_[2]:.2f}, * ubicacion")
joblib.dump(modelo, 'modelo_ventas_tiendas.pkl')
joblib.dump(x.columns.to_list(), 'columnas_ventas_tiendas.pkl')