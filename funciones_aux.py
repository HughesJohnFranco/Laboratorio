import cv2
import numpy as np
from collections import deque
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def normalizar_keypoints(keypoints):
    if len(keypoints) != 99:
        return keypoints
    keypoints = np.array(keypoints).reshape((33, 3))
    centro = (keypoints[23] + keypoints[24]) / 2
    keypoints -= centro

    hombro_izq = keypoints[11]
    hombro_der = keypoints[12]
    escala = np.linalg.norm(hombro_izq - hombro_der)
    if escala > 0:
        keypoints /= escala

    return keypoints.flatten()

def extraer_keypoints(resultados):
    keypoints = []
    if resultados.pose_landmarks:
        for lm in resultados.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 33 * 3)
    if resultados.left_hand_landmarks:
        for lm in resultados.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 21 * 3)
    if resultados.right_hand_landmarks:
        for lm in resultados.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 21 * 3)
    return normalizar_keypoints(keypoints)

def dibujar_keypoints(imagen, resultados):
    mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(imagen, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(imagen, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def mediapipe_deteccion(imagen, modelo):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_rgb.flags.writeable = False
    resultados = modelo.process(imagen_rgb)
    imagen_rgb.flags.writeable = True
    return resultados

class Suavizador:
    def __init__(self, tamaño_ventana=5):
        self.ventana = deque(maxlen=tamaño_ventana)
    def suavizar(self, keypoints):
        self.ventana.append(keypoints)
        if len(self.ventana) < self.ventana.maxlen:
            return keypoints
        return np.mean(self.ventana, axis=0)

def extraer_keypoints_de_frame(frame, modelo):
    resultados = mediapipe_deteccion(frame, modelo)
    return extraer_keypoints(resultados)
