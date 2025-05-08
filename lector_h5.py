import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Cargar el conjunto de datos
datos = pd.read_hdf('dataset_lenguaje.h5', key='data')

# Mostrar una vista previa y el tamaño del dataset
print(datos.head())
print(datos.shape)

# Convertir la columna de puntos clave a un array de NumPy
entradas = np.array(datos['keypoints'].tolist())  # Forma: (ejemplos, cuadros, características)
etiquetas = datos['label'].values

# Codificar etiquetas (de texto a números)
codificador = LabelEncoder()
etiquetas_codificadas = codificador.fit_transform(etiquetas)
etiquetas_onehot = tf.keras.utils.to_categorical(etiquetas_codificadas)

# Dividir en datos de entrenamiento y validación
x_entrenamiento, x_validacion, y_entrenamiento, y_validacion = train_test_split(
    entradas, etiquetas_onehot, test_size=0.2, random_state=42
)

# Verificar la forma final de los datos
print("Forma de x_entrenamiento:", x_entrenamiento.shape)

# Definir el modelo con capas LSTM
modelo = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(x_entrenamiento.shape[1], x_entrenamiento.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(etiquetas_onehot.shape[1], activation='softmax')  # Capa de salida con tantas neuronas como clases
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.summary()

# Entrenar el modelo
modelo.fit(x_entrenamiento, y_entrenamiento, epochs=30, validation_data=(x_validacion, y_validacion), batch_size=32)

# Guardar el modelo entrenado
modelo.save("modelo_lenguaje.keras")
