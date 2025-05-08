import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import dibujar_keypoints, mediapipe_deteccion
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo
modelo = tf.keras.models.load_model('modelo_red_neuronal.keras')

# Cargar el codificador de etiquetas
etiquetas = ["bien", "hola"]  # Asegúrate de que sean las mismas etiquetas usadas en el entrenamiento
encoder = LabelEncoder()
encoder.fit(etiquetas)

def preprocesar_keypoints(resultados):
    """ Extrae keypoints y los convierte en un array numpy adecuado para la predicción """
    if resultados.pose_landmarks:
        keypoints = np.array([[kp.x, kp.y, kp.z] for kp in resultados.pose_landmarks.landmark]).flatten()
        if keypoints.shape[0] != 99:  # Asegura que coincida con la entrada del modelo
            return None
        return keypoints
    return None

def detector_completo():
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Detección con Mediapipe
            resultados = mediapipe_deteccion(frame, holistic_model)

            # Extraer keypoints
            keypoints = preprocesar_keypoints(resultados)
            etiqueta_predicha = "Desconocido"  # Valor por defecto

            if keypoints is not None:
                keypoints = np.expand_dims(keypoints, axis=0)  # Adaptar para la entrada del modelo
                prediccion = modelo.predict(keypoints)
                etiqueta_predicha = encoder.inverse_transform([np.argmax(prediccion)])[0]
                print(f"Gesto detectado: {etiqueta_predicha}")  # Mostrar el gesto reconocido

            # Dibujar keypoints
            imagen = frame.copy()
            dibujar_keypoints(imagen, resultados)

            # Mostrar el nombre del gesto en la esquina superior izquierda
            cv2.putText(imagen, f"Gesto: {etiqueta_predicha}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostrar en pantalla
            cv2.imshow('Detector de gestos', imagen)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector_completo()
