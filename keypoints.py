# aca generamos los .h5
# con esto generamos la carpeta data donde se guardan los keypoints

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

palabras = ['bien', 'hola', 'Buenas_Noches','SinGesto']
ruta_base = 'C:\\Users\\alen_\\repos\\Laboratorio\\Laboratorio\\frame_acciones'
ruta_salida = 'C:\\Users\\alen_\\repos\\Laboratorio\\Laboratorio\\data\\keypoints' 

os.makedirs(ruta_salida, exist_ok=True)

for palabra in palabras:
    datos = []
    ruta_palabra = os.path.join(ruta_base, palabra)

    for carpeta_raiz, _, archivos in os.walk(ruta_palabra):
        for archivo in archivos:
            if not archivo.endswith('.jpg'):
                continue
            ruta_imagen = os.path.join(carpeta_raiz, archivo)
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                continue
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            resultados = pose.process(imagen_rgb)

            if resultados.pose_landmarks:
                keypoints = []
                for lm in resultados.pose_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                if len(keypoints) == 99:
                    datos.append(keypoints)


    columnas = [f"{i}_{coord}" for i in range(33) for coord in ['x', 'y', 'z']]
    df = pd.DataFrame(datos, columns=columnas)    
    archivo_salida = os.path.join(ruta_salida, f"{palabra}.h5")
    df.to_hdf(archivo_salida, key='data', mode='w')
    print(f"Guardado: {archivo_salida}")        
        
