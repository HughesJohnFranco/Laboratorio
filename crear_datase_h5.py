import os
import numpy as np
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
import cv2
from zhelpers import mediapipe_detection, extract_keypoints
from tqdm import tqdm #barra de progesssssshjjhjhjhjhjh ho

def create_dataset_h5(base_path, output_path="dataset.h5"):
    data = []
    labels =[]
    labels_map= {}

    with Holistic(static_image_mode=True) as model:
        palabras = os.listdir(base_path)
        for idx, palabra in enumerate(palabras):
            palabra_path = os.path.join(base_path, palabra)
            if not os.path.isdir(palabra_path):
                continue
            labels_map[idx] = palabra
            muestra = os.listdir(palabra_path)

            for muestra in tqdm(muestra, desc=f"Procesando {palabra}"):
                muestra_path = os.path.join(palabra_path, muestra)
                frames = sorted(os.listdir(muestra_path))
                secuencia = []

                for frame_name in frames:
                    frame_path = os.path.join(muestra_path, frame_name)
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        continue

                    results = mediapipe_detection(frame, model)
                    keypoints = extract_keypoints(results)
                    print(f"Frame procesado: {frame_name}, Keypoints extraidos: {len(keypoints)}")
                    secuencia.append(keypoints)
                    if frame is None:
                        print(f"No se pudo leer la imagen: {frame_path}")

                if len(secuencia) > 0:
                    data.append(secuencia)
                    labels.append(idx)

    df = pd.DataFrame({'keypoints': data, 'label': labels})
    df.to_hdf(output_path, key='data', mode='w')
    print(f"Data set guardado en {output_path}")
    print(f"Etiquetas:  {labels_map}")
    return labels_map

if __name__ == "__main__":
    create_dataset_h5("frame_actions","dataset_lenguaje.h5")