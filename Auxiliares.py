import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

# Detecta los puntos clave con MediaPipe
def deteccion_mediapipe(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    resultados = modelo.process(imagen)
    return resultados

# Crea una carpeta si no existe
def crear_carpeta(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

# Verifica si hay alguna mano detectada
def hay_mano(resultados):
    return resultados.left_hand_landmarks or resultados.right_hand_landmarks

# Dibuja los puntos clave en la imagen
def dibujar_puntos_clave(imagen, resultados):
    if resultados.face_landmarks:
        draw_landmarks(
            imagen,
            resultados.face_landmarks,
            FACEMESH_CONTOURS,
            DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if resultados.pose_landmarks:
        draw_landmarks(
            imagen,
            resultados.pose_landmarks,
            POSE_CONNECTIONS,
            DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if resultados.left_hand_landmarks:
        draw_landmarks(
            imagen,
            resultados.left_hand_landmarks,
            HAND_CONNECTIONS,
            DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if resultados.right_hand_landmarks:
        draw_landmarks(
            imagen,
            resultados.right_hand_landmarks,
            HAND_CONNECTIONS,
            DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

# Guarda cada frame como imagen en la carpeta destino
def guardar_frame(cuadros, carpeta_salida):
    for numero_frame, cuadro in enumerate(cuadros):
        ruta_cuadro = os.path.join(carpeta_salida, f"{numero_frame + 1}.jpg")
        cv2.imwrite(ruta_cuadro, cv2.cvtColor(cuadro, cv2.COLOR_BGR2BGRA))

# Extrae los puntos clave de pose, cara y manos como un solo vector
def extraer_keypoints(resultados):
    pose = np.array([[p.x, p.y, p.z, p.visibility] for p in resultados.pose_landmarks.landmark]).flatten() if resultados.pose_landmarks else np.zeros(33*4)
    cara = np.array([[p.x, p.y, p.z] for p in resultados.face_landmarks.landmark]).flatten() if resultados.face_landmarks else np.zeros(468*3)
    mano_izq = np.array([[p.x, p.y, p.z] for p in resultados.left_hand_landmarks.landmark]).flatten() if resultados.left_hand_landmarks else np.zeros(21*3)
    mano_der = np.array([[p.x, p.y, p.z] for p in resultados.right_hand_landmarks.landmark]).flatten() if resultados.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, cara, mano_izq, mano_der])
