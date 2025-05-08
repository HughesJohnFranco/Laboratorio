import os
import numpy as np
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
import cv2
from Auxiliares import deteccion_mediapipe, extraer_keypoints
from tqdm import tqdm  # barra de progreso

def crear_h5(ruta_base, salida="dataset.h5"):
    datos = []
    etiquetas = []
    mapa_etiquetas = {}

    with Holistic(static_image_mode=True) as modelo:
        palabras = os.listdir(ruta_base)
        for indice, palabra in enumerate(palabras):
            ruta_palabra = os.path.join(ruta_base, palabra)
            if not os.path.isdir(ruta_palabra):
                continue

            mapa_etiquetas[indice] = palabra
            muestras = os.listdir(ruta_palabra)

            for muestra in tqdm(muestras, desc=f"Procesando {palabra}"):
                ruta_muestra = os.path.join(ruta_palabra, muestra)
                fotogramas = sorted(os.listdir(ruta_muestra))
                secuencia = []

                for nombre_frame in fotogramas:
                    ruta_fotograma = os.path.join(ruta_muestra, nombre_frame)
                    imagen = cv2.imread(ruta_fotograma)
                    if imagen is None:
                        print(f"No se pudo leer la imagen: {ruta_fotograma}")
                        continue

                    resultado = deteccion_mediapipe(imagen, modelo)
                    puntos_clave = extraer_keypoints(resultado)
                    print(f"Fotograma procesado: {nombre_frame}, Puntos clave extraídos: {len(puntos_clave)}")
                    secuencia.append(puntos_clave)

                if len(secuencia) > 0:
                    datos.append(secuencia)
                    etiquetas.append(indice)

    df = pd.DataFrame({'keypoints': datos, 'label': etiquetas})
    df.to_hdf(salida, key='data', mode='w')
    print(f"Dataset guardado en: {salida}")
    print(f"Etiquetas asignadas: {mapa_etiquetas}")
    return mapa_etiquetas

if __name__ == "__main__":
    crear_h5("frame_actions", "dataset_lenguaje.h5")
