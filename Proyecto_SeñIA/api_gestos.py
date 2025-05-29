from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import time
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.hands import Hands
from funciones_aux import mediapipe_deteccion
from sklearn.preprocessing import LabelEncoder
from threading import Lock

app = Flask(__name__)

# Cargo la red neuronal y las etiquetas que puede predecir para luego compararlos...
modelo = tf.keras.models.load_model('modelo_red_neuronal.keras')
etiquetas = ["bien", "chill", "fuckyou", "hola", "mal", "mate", "okey", "paz", "te_amo"]
encoder = LabelEncoder()
encoder.fit(etiquetas)

# Variables compartidas
ultima_etiqueta = None
etiqueta_actual = "..."
lock = Lock()


# Procesar keypoints
def preprocesar_keypoints(resultados):
    if resultados.pose_landmarks:
        keypoints = np.array([[kp.x, kp.y, kp.z] for kp in resultados.pose_landmarks.landmark]).flatten()
        if len(keypoints) == 99:
            return keypoints
    return None

def hay_manos(resultados_manos):
    return resultados_manos.multi_hand_landmarks is not None


# Activo la cámara, detecto con mediapipe y preproceso los keypoints para pasárselos al modelo keras
def generar_video():
    global ultima_etiqueta, etiqueta_actual

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠ No se pudo abrir la cámara.")
        return

    tiempo_ultima_actualizacion = 0  # delay entre gestos

    with Holistic() as holistic, Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as manos:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resultados_holistic = mediapipe_deteccion(frame, holistic)
            resultados_manos = manos.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            etiqueta = "..."

            # Si se detectan manos, se predice el gesto con el modelo keras
            if hay_manos(resultados_manos):
                keypoints = preprocesar_keypoints(resultados_holistic)
                if keypoints is not None:
                    entrada = np.expand_dims(keypoints, axis=0)
                    prediccion = modelo.predict(entrada)
                    etiqueta = encoder.inverse_transform([np.argmax(prediccion)])[0]

            # Mostrar texto en pantalla
            cv2.putText(frame, f"Gesto: {etiqueta}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Solo actualizar si pasaron al menos 2 segundos desde el último cambio
            tiempo_actual = time.time()
            with lock:
                if (
                    etiqueta != ultima_etiqueta
                    and etiqueta in etiquetas
                    and (tiempo_actual - tiempo_ultima_actualizacion > 4)
                ):
                    etiqueta_actual = etiqueta
                    ultima_etiqueta = etiqueta
                    tiempo_ultima_actualizacion = tiempo_actual

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# Rutas de Flask
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generar_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_gesto')
def get_gesto():
    with lock:
        return jsonify({'gesto': etiqueta_actual})


if __name__ == '__main__':
    app.run(debug=True)
