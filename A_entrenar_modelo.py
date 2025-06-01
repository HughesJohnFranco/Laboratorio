import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# --- Configuración ---
RUTA_KEYPOINTS = 'C:/Users/alen_/repos/Laboratorio/data/keypoints'

X = []
y = []

# --- Cargar archivos ---
print("📂 Cargando secuencias desde:", RUTA_KEYPOINTS)
for archivo in os.listdir(RUTA_KEYPOINTS):
    if not archivo.endswith('.h5'):
        continue

    ruta_archivo = os.path.join(RUTA_KEYPOINTS, archivo)
    df = pd.read_hdf(ruta_archivo)

    if df.shape != (10, 225):  # ✅ Ajuste aquí
        print(f"❌ Formato inválido: {archivo} -> {df.shape}")
        continue

    X.append(df.values)

    # Etiqueta: todo menos el "_muestraX.h5"
    etiqueta = '_'.join(archivo.split('_')[:-1])
    y.append(etiqueta)

print("✅ Cargadas muestras:", len(X))

# --- Verificar clases ---
print("\n📊 Distribución de clases:")
conteo = Counter(y)
for clase, cantidad in conteo.items():
    print(f"  {clase}: {cantidad} muestras")

# --- Preparar datos ---
X = np.array(X)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\n🌤 Clases codificadas:")
for clase, idx in zip(encoder.classes_, range(len(encoder.classes_))):
    print(f"  {clase}: {idx}")

# --- Dividir en entrenamiento y testeo ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.33, random_state=42, stratify=y_encoded
)

print(f"\n📁 Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

# --- Definir modelo ---
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10, 225)),  # ✅ Ajuste aquí también
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Entrenar ---
print("\n🚀 Entrenando modelo...")
hist = modelo.fit(X_train, y_train, epochs=250, batch_size=4, validation_data=(X_test, y_test))

# --- Evaluación ---
print("\n📈 Evaluando modelo...")
loss, acc = modelo.evaluate(X_test, y_test)
print(f"\n✅ Precisión del modelo: {acc:.2f}")

# --- Diagnóstico de predicciones ---
print("\n🔎 Diagnóstico de predicciones:")
predicciones = modelo.predict(X_test)
clases_predichas = np.argmax(predicciones, axis=1)

print("🟡 Clases verdaderas: ", y_test)
print("🔹 Clases predichas:  ", clases_predichas)
print("🌤 Etiquetas reales:  ", encoder.inverse_transform(y_test))
print("🌤 Etiquetas predichas:", encoder.inverse_transform(clases_predichas))

# --- Guardar modelo ---
modelo.save('modelo_secuencial_lsa.keras')
print("\n📂 Modelo guardado como 'modelo_secuencial_lsa.keras'")
