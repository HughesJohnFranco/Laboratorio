# este detector no dibuja los puntos, pero si traduce (consume menos recursos)

import cv2
import numpy as np
import tensorflow as tf
import pygame
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.hands import Hands
from funciones_aux import mediapipe_deteccion
from sklearn.preprocessing import LabelEncoder

pygame.mixer.init()  # Inicializador para los sonidos

# Cargar sonidos
sonidos = {
    "hola": pygame.mixer.Sound("Laboratorio/audio/hola.wav"),
    "bien": pygame.mixer.Sound("Laboratorio/audio/bien.wav"),
    "mal": pygame.mixer.Sound("Laboratorio/audio/mal.wav"),
    "okey": pygame.mixer.Sound("Laboratorio/audio/okey.wav")
}

# Cargar el modelo de clasificación de gestos
modelo = tf.keras.models.load_model('modelo_red_neuronal_actualizado.keras')

# Codificador de etiquetas
etiquetas = ["bien", "hola", "mal", "okey"]
encoder = LabelEncoder()
encoder.fit(etiquetas)

def preprocesar_keypoints(resultados):
    """ Extrae keypoints y los convierte en un array numpy adecuado para la predicción """
    if resultados.pose_landmarks:
        keypoints = np.array([[kp.x, kp.y, kp.z] for kp in resultados.pose_landmarks.landmark]).flatten()
        if keypoints.shape[0] != 99:
            return None
        return keypoints
    return None

def hay_manos(resultados_manos):
    """ Verifica si hay detección de manos en la imagen """
    return resultados_manos.multi_hand_landmarks is not None

#def dibujar_puntos(imagen, resultados):
#     Dibuja solo los puntos clave sin líneas de conexión """
#    if resultados.pose_landmarks:
#        for kp in resultados.pose_landmarks.landmark:
#            x, y = int(kp.x * imagen.shape[1]), int(kp.y * imagen.shape[0])
#            cv2.circle(imagen, (x, y), 3, (0, 255, 0), -1)  # Dibuja solo los puntos verdes

def detector_completo():
    ultima_etiqueta = None  # Para evitar repetir el sonido constantemente

    try:
        video = cv2.VideoCapture(1)  # Se usa la cámara predeterminada

        if not video.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

        with Holistic() as holistic_model, Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as manos_model:
            while True:
                ret, frame = video.read()
                if not ret or frame is None:
                    print("No se recibió un frame, cerrando...")
                    break

                resultados_holistic = mediapipe_deteccion(frame, holistic_model)
                resultados_manos = manos_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                etiqueta_predicha = "Desconocido"

                # Solo detectar gestos si hay manos presentes
                if hay_manos(resultados_manos):
                    keypoints = preprocesar_keypoints(resultados_holistic)
                    if keypoints is not None:
                        keypoints = np.expand_dims(keypoints, axis=0)
                        prediccion = modelo.predict(keypoints)
                        etiqueta_predicha = encoder.inverse_transform([np.argmax(prediccion)])[0]

                        # Reproducir sonido si es un gesto nuevo
                        if etiqueta_predicha != ultima_etiqueta and etiqueta_predicha in sonidos:
                            sonidos[etiqueta_predicha].play()
                            ultima_etiqueta = etiqueta_predicha

                        print(f"Gesto detectado: {etiqueta_predicha}")
                    else:
                        etiqueta_predicha = "Sin detección válida"
                else:
                    etiqueta_predicha = "..."

                imagen = frame.copy()
                #dibujar_puntos(imagen, resultados_holistic)  # Solo dibujamos los puntos

                cv2.putText(imagen, f"Gesto: {etiqueta_predicha}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Detector de gestos', imagen)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        video.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error detectado: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    detector_completo()
