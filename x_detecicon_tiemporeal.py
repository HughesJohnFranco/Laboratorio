import cv2
import numpy as np
import joblib
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import dibujar_keypoints, mediapipe_deteccion

# Cargar el modelo entrenado
modelo = joblib.load('modelo_randomforest.pkl')

# Función para extraer keypoints
def extraer_keypoints(resultados):
    pose = []
    if resultados.pose_landmarks:
        pose = [coord for punto in resultados.pose_landmarks.landmark for coord in (punto.x, punto.y, punto.z)]
    else:
        pose = [0] * 99  # 33 puntos * 3 coordenadas
    return pose

def detector_completo():
    with Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.6) as holistic_model:
        video = cv2.VideoCapture(1)  # cambiar a 0 si la cámara no funciona

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            resultados = mediapipe_deteccion(frame, holistic_model)
            imagen = frame.copy()

            dibujar_keypoints(imagen, resultados)

            # Extraer keypoints y predecir
            keypoints = extraer_keypoints(resultados)
            X = np.array(keypoints).reshape(1, -1)
            prediccion = modelo.predict(X)[0]
            probabilidad = modelo.predict_proba(X).max()

            # Mostrar texto con predicción
            cv2.putText(imagen, f'{prediccion} ({probabilidad:.2f})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Detector + Reconocimiento', imagen)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector_completo()
