import cv2
import numpy as np
import tensorflow as tf
import pygame
import time
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import dibujar_keypoints, mediapipe_deteccion
from sklearn.preprocessing import LabelEncoder

# Inicializar sonidos
pygame.mixer.init()

sonidos = {
    "hola": pygame.mixer.Sound("C:/Users/alen_/repos/Laboratorio/Laboratorio/audio/hola.wav"),
    "bien": pygame.mixer.Sound("C:/Users/alen_/repos/Laboratorio/Laboratorio/audio/bien.wav"),
    "mal": pygame.mixer.Sound("C:/Users/alen_/repos/Laboratorio/Laboratorio/audio/mal.wav"),
    "okey": pygame.mixer.Sound("C:/Users/alen_/repos/Laboratorio/Laboratorio/audio/okey.wav"),
    "Buenas_Noches": pygame.mixer.Sound("C:/Users/alen_/repos/Laboratorio/Laboratorio/audio/buenasnoches.wav")
}

# Cargar modelo
modelo = tf.keras.models.load_model('modelo_red_neuronal.keras')

# Cargar etiquetas
etiquetas = ["bien", "hola", "mal", "okey", "Buenas_Noches"]
encoder = LabelEncoder()
encoder.fit(etiquetas)

def preprocesar_keypoints(resultados):
    if resultados.pose_landmarks:
        keypoints = np.array([[kp.x, kp.y, kp.z] for kp in resultados.pose_landmarks.landmark]).flatten()
        if keypoints.shape[0] != 99:
            return None
        return keypoints
    return None

def detector_completo():
    ultima_etiqueta = None
    etiqueta_repetida = False
    tiempo_ultima_det = 1
    delay_segundos = 3

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

                tiempo_actual = time.time()

                # Si cambia la seña, permitir que se reproduzca de nuevo
                if etiqueta_predicha != ultima_etiqueta:
                    etiqueta_repetida = False

                if (not etiqueta_repetida and etiqueta_predicha in sonidos and
                    etiqueta_predicha != "Desconocido"):

                    if tiempo_actual - tiempo_ultima_det >= delay_segundos:
                        sonidos[etiqueta_predicha].play()
                        tiempo_ultima_det = tiempo_actual
                        ultima_etiqueta = etiqueta_predicha
                        etiqueta_repetida = True

                print(f"Gesto detectado: {etiqueta_predicha}")

            else:
                etiqueta_predicha = "Desconocido"

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