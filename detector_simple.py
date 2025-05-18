import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import dibujar_keypoints, mediapipe_deteccion

def detector_completo():

    with Holistic() as holistic_model:
        
        video = cv2.VideoCapture(1)
        while video.isOpened():

            ret, frame = video.read()
            if not ret:
                break
            
            # detecion con meidpipe
            resultados = mediapipe_deteccion(frame, holistic_model)

            # copia del frame para dibujar
            imagen = frame.copy()

            # Dibujar keypoints
            dibujar_keypoints(imagen, resultados)

            # motrar en pantalla
            cv2.imshow('Solo_detector', imagen)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    detector_completo()