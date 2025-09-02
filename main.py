from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

model = joblib.load("modelo_ventas_tiendas.pkl")
columns = joblib.load("columnas_ventas_tiendas.pkl")

app = FastAPI(tittle="Prediccion precio de ventas")

#modelo de entrada

class InputData(BaseModel):
    empleados: float
    publicidad: float
    ubicacion: int

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de ventas"}

@app.post("/predict")
def predict(data: InputData):
    x_new = pd.DataFrame([[data.empleados, data.publicidad, data.ubicacion]], columns=columns)
    prediccion = model.predict(x_new)
    return {"prediccion": float(prediccion[0])}
