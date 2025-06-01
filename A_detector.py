import cv2
import numpy as np
import tensorflow as tf
import time
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import dibujar_keypoints, mediapipe_deteccion, extraer_keypoints
from sklearn.preprocessing import LabelEncoder
from collections import deque

modelo = tf.keras.models.load_model('modelo_secuencial_lsa.keras')
etiquetas = ["bien", "chill", "fuckyou", "hola","mal","mate", "okey", "paz","teamo"]
encoder = LabelEncoder()
encoder.fit(etiquetas)

def extraer_keypoints(resultados):
    def extraer_landmarks(landmarks, cantidad):
        if landmarks:
            return np.array([[kp.x, kp.y, kp.z] for kp in landmarks.landmark]).flatten()
        else:
            return np.zeros(cantidad * 3)

    pose = extraer_landmarks(resultados.pose_landmarks, 33)
    mano_izquierda = extraer_landmarks(resultados.left_hand_landmarks, 21)
    mano_derecha = extraer_landmarks(resultados.right_hand_landmarks, 21)

    return np.concatenate([pose, mano_izquierda, mano_derecha])

def detector_completo():
    buffer_frames = deque(maxlen=10)
    tiempo_espera = None
    etiqueta_predicha = ""
    tiempo_mostrar_etiqueta = 0

    with Holistic(static_image_mode=False, model_complexity=2,
                  min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic_model:
        video = cv2.VideoCapture(1)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            resultados = mediapipe_deteccion(frame, holistic_model)
            keypoints = extraer_keypoints(resultados)

            if keypoints is not None:
                buffer_frames.append(keypoints)

            if len(buffer_frames) == 10 and tiempo_espera is None:
                tiempo_espera = time.time()

            if tiempo_espera is not None and time.time() - tiempo_espera >= 3:
                secuencia = np.array(buffer_frames).reshape(1, 10, 225)
                prediccion = modelo.predict(secuencia, verbose=0)
                etiqueta_predicha = encoder.inverse_transform([np.argmax(prediccion)])[0]
                tiempo_mostrar_etiqueta = time.time() + 5
                tiempo_espera = None
                buffer_frames.clear()

            imagen = frame.copy()
            dibujar_keypoints(imagen, resultados)

            if time.time() < tiempo_mostrar_etiqueta:
                cv2.putText(imagen, f"Gesto: {etiqueta_predicha}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Detector de gestos en vivo', imagen)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector_completo()
