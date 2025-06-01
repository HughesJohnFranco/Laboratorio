import os
import cv2
import numpy as np
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from funciones_aux import crear_carpeta, dibujar_keypoints, mediapipe_deteccion, guardar_frames, hay_manos
from constantes import FUENTE, FUENTE_POS, TAMANIO_FUENTE, FRAME_RUTA_ACCIONES, RUTA_RAIZ

def capture_samples(ruta, margen_frame=1, min_cant_frames=10, delay_frames=3):
    crear_carpeta(ruta)
    
    contar_frame = 0
    frames = []
    arrg_frames = 0
    grabando = False
    
    with Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
        video = cv2.VideoCapture(1)  # Asegurate que el índice sea correcto (0 o 1 según tu cámara)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            imagen = frame.copy()
            resultados = mediapipe_deteccion(frame, holistic_model)

            if hay_manos(resultados) or grabando:
                grabando = False
                contar_frame += 1
                if contar_frame > margen_frame:
                    cv2.putText(imagen, 'Capturando...', FUENTE_POS, FUENTE, TAMANIO_FUENTE, (255, 50, 0))
                    frames.append(frame)
            else:
                if len(frames) >= min_cant_frames + margen_frame:
                    arrg_frames += 1
                    if arrg_frames < delay_frames:
                        grabando = True
                        continue
                    frames = frames[: - (margen_frame + delay_frames)]
                    timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
                    ruta_salida = os.path.join(ruta, f"sample_{timestamp}")
                    crear_carpeta(ruta_salida)
                    guardar_frames(frames, ruta_salida)

                # Reset para siguiente captura
                grabando, arrg_frames = False, 0
                frames, contar_frame = [], 0
                cv2.putText(imagen, 'Listo para capturar...', FUENTE_POS, FUENTE, TAMANIO_FUENTE, (0, 220, 100))

            dibujar_keypoints(imagen, resultados)
            cv2.imshow(f'Toma de muestras: "{os.path.basename(ruta)}"', imagen)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    palabra_nueva = "buenas_noches"
    ruta_guardado = os.path.join(FRAME_RUTA_ACCIONES, palabra_nueva)
    capture_samples(ruta_guardado)
