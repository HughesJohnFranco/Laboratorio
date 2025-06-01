import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
from funciones_aux import extraer_keypoints_de_frame, Suavizador, dibujar_keypoints, mediapipe_deteccion

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2,
                                 min_detection_confidence=0.7, min_tracking_confidence=0.7)

palabras = ["bien", "chill", "fuckyou", "hola", "mal", "mate", "okey", "paz","teamo"]
ruta_base = "C:/Users/alen_/repos/Laboratorio/frame_acciones"
ruta_salida = 'C:/Users/alen_/repos/Laboratorio/data/keypoints'
os.makedirs(ruta_salida, exist_ok=True)

suavizador = Suavizador(tamaño_ventana=5)

for palabra in palabras:
    ruta_palabra = os.path.join(ruta_base, palabra)
    for muestra in os.listdir(ruta_palabra):
        ruta_muestra = os.path.join(ruta_palabra, muestra)
        keypoints_secuencia = []
        for i in range(1, 11):
            ruta_imagen = os.path.join(ruta_muestra, f"{i}.jpg")
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"Imagen no encontrada: {ruta_imagen}")
                continue

            resultados = mediapipe_deteccion(imagen, holistic)
            dibujar_keypoints(imagen, resultados)
            cv2.imshow("Keypoints", imagen)
            cv2.waitKey(100)

            keypoints = extraer_keypoints_de_frame(imagen, holistic)
            keypoints = suavizador.suavizar(keypoints)
            keypoints_secuencia.append(keypoints)

        if len(keypoints_secuencia) == 10:
            df = pd.DataFrame(keypoints_secuencia)
            nombre_archivo = f"{palabra}_{muestra}.h5"
            archivo_salida = os.path.join(ruta_salida, nombre_archivo)
            df.to_hdf(archivo_salida, key='data', mode='w')
            print(f"✅ Guardado: {archivo_salida}")
        else:
            print(f"❌ Muestra inválida: {palabra}/{muestra} (frames válidos: {len(keypoints_secuencia)})")

cv2.destroyAllWindows()
