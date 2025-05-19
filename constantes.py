
import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 15

# PATHS
RUTA_RAIZ = os.getcwd()
FRAME_RUTA_ACCIONES = os.path.join(RUTA_RAIZ, "frame_acciones")
DATA_PATH = os.path.join(RUTA_RAIZ, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_FOLDER_PATH = os.path.join(RUTA_RAIZ, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# MOSTRAR PARÁMETROS DE IMAGEN
FUENTE = cv2.FONT_HERSHEY_PLAIN
TAMANO_FUENTE = 1.5
FUENTE_POS = (5, 30)


# Palabras que estamos traduciendo (solo estas por ahora)
texto_palabras = {
    "hola": "HOLA",
    "como_estas": "COMO ESTÁS",
    "bien": "BIEN",
    "mal": "MAL",
    "gracias": "GRACIAS",
    "de_nada": "DE NADA",
    "adios": "ADIÓS",
    "Buenas_Noches":"buenasnoches",
    "SinGesto": "Singesto"
}


# constantes.py (adaptado)
import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 15

# PATHS
actions = ['hola', 'bien', 'Buenas_Noches','SinGesto']
RUTA_RAIZ = os.getcwd()
FRAME_RUTA_ACCIONES = os.path.join(RUTA_RAIZ, 'frame_acciones')
DATA_PATH = os.path.join(RUTA_RAIZ, 'data')
KEYPOINTS_PATH = os.path.join(DATA_PATH, 'keypoints')
MODEL_FOLDER_PATH = os.path.join(RUTA_RAIZ, 'models')
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f'actions_{MODEL_FRAMES}.keras')
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, 'words.json')

# FONT SETTINGS
FUENTE = cv2.FONT_HERSHEY_PLAIN
TAMANIO_FUENTE = 1.5
FUENTE_POS = (5, 30)