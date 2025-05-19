import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import crear_carpeta, dibujar_keypoints, mediapipe_deteccion, guardar_frames, hay_manos
from constantes import FUENTE, FUENTE_POS, TAMANO_FUENTE, FRAME_RUTA_ACCIONES, RUTA_RAIZ
from datetime import datetime


def capture_samples(ruta, margen_frame=1, min_cant_frames=10, delay_frames=3):

    crear_carpeta(ruta)
    
    contar_frame = 0
    frames = []
    arrg_frames = 0
    grabacion = True
    
    with Holistic() as holistic_model:

        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            imagen = frame.copy()
            resultados = mediapipe_deteccion(frame, holistic_model)
            
            if hay_manos(resultados) or grabacion:
                grabacion = False
                contar_frame += 1
                if contar_frame > margen_frame:
                    cv2.putText(imagen, 'Capturando...', FUENTE_POS, FUENTE, TAMANO_FUENTE, (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) >= min_cant_frames + margen_frame:
                    arrg_frames += 1
                    if arrg_frames < delay_frames:
                        grabacion = True
                        continue
                    frames = frames[: - (margen_frame + delay_frames)]
                    el_dia = datetime.now().strftime('%y%m%d%H%M%S%f')
                    carpeta_de_salida = os.path.join(ruta, f"sample_{el_dia}")
                    crear_carpeta(carpeta_de_salida)
                    guardar_frames(frames, carpeta_de_salida)
                
                grabacion, arrg_frames = False, 0
                frames, contar_frame = [], 0
                cv2.putText(imagen, 'Listo para capturar...', FUENTE_POS, FUENTE, TAMANO_FUENTE, (0,220, 100))
            

            dibujar_keypoints(imagen, resultados)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(ruta)}"', imagen)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    nombre_palabra = "SinGesto"
    ruta_palabra = os.path.join(RUTA_RAIZ, FRAME_RUTA_ACCIONES, nombre_palabra)
    capture_samples(ruta_palabra)