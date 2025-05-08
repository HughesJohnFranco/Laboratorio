# Este archivo lo ejecutan despues de obtener el archivo .h5
# Es la RED NEURONAL

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ruta de los archivos .h5 generados
ruta_keypoints = 'C:/Users/John/Desktop/LABORATORIO_2025/data/keypoints'
archivos = [f for f in os.listdir(ruta_keypoints) if f.endswith('.h5')]

X = []
y = []

for archivo in archivos:
    ruta = os.path.join(ruta_keypoints, archivo)
    df = pd.read_hdf(ruta)
    
    # Si el archivo no tiene columnas esperadas, saltealo
    if df.empty or df.shape[1] != 99:
        print(f"Archivo inválido: {archivo}, columnas: {df.shape[1]}")
        continue
    
    datos = df.values  # convierte el DataFrame en un array de numpy
    X.extend(datos)    # agregamos cada fila como un vector
    etiqueta = archivo.split('.')[0]  # 'bien', 'hola', etc.
    y.extend([etiqueta] * len(datos))  # etiquetas para cada muestra

X = np.array(X)
y = np.array(y)

# División de datos en entrenamiento y test
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)

# Evaluar
precision = modelo.score(X_prueba, y_prueba)
print(f"Precisión del modelo: {precision:.2f}")

# Guardar modelo
joblib.dump(modelo, 'modelo_randomforest.pkl')
print("Modelo guardado como 'modelo_randomforest.pkl'")
