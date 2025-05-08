import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ruta de los archivos .h5 generados
ruta_keypoints = 'C:/Users/John/Desktop/LAB_SACACHISPAS/Laboratorio/data/keypoints'
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

# Codificar las etiquetas
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# División de datos en entrenamiento y test
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Construcción del modelo en TensorFlow
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_entrenamiento.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')  # Salida para clasificación
])

# Compilación del modelo
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
modelo.fit(X_entrenamiento, y_entrenamiento, epochs=50, batch_size=32, validation_data=(X_prueba, y_prueba))

# Evaluación
loss, precision = modelo.evaluate(X_prueba, y_prueba)
print(f"Precisión del modelo: {precision:.2f}")

# Guardar modelo en formato .keras
modelo.save('modelo_red_neuronal.keras')
print("Modelo guardado como 'modelo_red_neuronal.keras'")
