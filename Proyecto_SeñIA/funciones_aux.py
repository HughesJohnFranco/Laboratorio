import cv2
import mediapipe as mp

def mediapipe_deteccion(imagen, modelo):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = modelo.process(imagen_rgb)
    return resultados
