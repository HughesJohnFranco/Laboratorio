import cv2
import numpy as np
import tensorflow as tf
import pygame
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import dibujar_keypoints, mediapipe_deteccion
from sklearn.preprocessing import LabelEncoder


pygame.mixer.init()# inicializador para lossonidos 

# cargar sonidos
sonidos = {
    "hola": pygame.mixer.Sound("Laboratorio/audio/hola.wav"),
    "bien": pygame.mixer.Sound("Laboratorio/audio/bien.wav"),
    "mal": pygame.mixer.Sound("Laboratorio/audio/mal.wav"),
    "okey": pygame.mixer.Sound("Laboratorio/audio/okey.wav")
}

# Cargar el modelo
modelo = tf.keras.models.load_model('modelo_red_neuronal_actualizado.keras')

# cargar el codificador de etiquetas
etiquetas = ["bien", "hola", "mal", "okey"]
encoder = LabelEncoder()
encoder.fit(etiquetas)

def preprocesar_keypoints(resultados):
    #extrae keypoints y los convierte en un array numpy adecuado para la predicción
    if resultados.pose_landmarks:
        keypoints = np.array([[kp.x, kp.y, kp.z] for kp in resultados.pose_landmarks.landmark]).flatten()
        if keypoints.shape[0] != 99:
            return None
        return keypoints
    return None

def detector_completo():
    ultima_etiqueta = None  # Para evitar repetir el sonido constantemente

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(1)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            resultados = mediapipe_deteccion(frame, holistic_model)
            keypoints = preprocesar_keypoints(resultados)
            etiqueta_predicha = "Desconocido"

            if keypoints is not None:
                keypoints = np.expand_dims(keypoints, axis=0)
                prediccion = modelo.predict(keypoints)
                etiqueta_predicha = encoder.inverse_transform([np.argmax(prediccion)])[0]

                # Reproducir sonido si es un gesto nuevo
                if etiqueta_predicha != ultima_etiqueta and etiqueta_predicha in sonidos:
                    sonidos[etiqueta_predicha].play()
                    ultima_etiqueta = etiqueta_predicha

                print(f"Gesto detectado: {etiqueta_predicha}")

            imagen = frame.copy()
            dibujar_keypoints(imagen, resultados)

            cv2.putText(imagen, f"Gesto: {etiqueta_predicha}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Detector de gestos', imagen)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector_completo()
