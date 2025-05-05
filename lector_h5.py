import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Cargar dataset
df = pd.read_hdf('dataset_lenguaje.h5', key='data')

# Verificá que tenga datos
print(df.head())
print(df.shape)

# Convertir columna 'keypoints' a array numpy
X = np.array(df['keypoints'].tolist())  # Esto da (samples, timesteps, features)
y = df['label'].values

# Codificar las etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Verificar forma final
print("Forma de X_train:", X_train.shape)  # Debería ser (n_samples, 30, features)

# Modelo LSTM
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)
model.save("modelo_lenguaje.keras")

