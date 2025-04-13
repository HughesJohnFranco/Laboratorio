import cv2
import mediapipe as mp
from config.constantes import MAX_MANOS, CONF_DETECCION, CONF_SEGUIMIENTO


class Mano:

    def __init__(self, mode=False, max_manos=MAX_MANOS, conf_deteccion=CONF_DETECCION, conf_segui=CONF_SEGUIMIENTO):
        self.mode = mode
        self.max_manos = max_manos
        self.conf_deteccion = conf_deteccion
        self.conf_segui = conf_segui

        self.mp_manos = mp.solutions.hands
        self.manos = self.mp_manos.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_manos,
            min_detection_confidence=self.conf_deteccion,
            min_tracking_confidence=self.conf_segui
        )
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]
        self.resultados = None


    def encontrar_manos(self, frame, dibujar=True):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(img_rgb)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(
                        frame, mano, self.mp_manos.HAND_CONNECTIONS,
                        self.dibujo.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                        self.dibujo.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
        return frame


    def encontrar_posicion(self, frame, mano_num=0, dibujar=True):
        x_lista, y_lista = [], []
        bbox = []
        lista = []

        if self.resultados and self.resultados.multi_hand_landmarks:
            mi_mano = self.resultados.multi_hand_landmarks[mano_num]

            alto, ancho, _ = frame.shape
            for id, lm in enumerate(mi_mano.landmark):
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                x_lista.append(cx)
                y_lista.append(cy)
                lista.append([id, cx, cy])

                if dibujar:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

            if x_lista and y_lista:
                xmin, xmax = min(x_lista), max(x_lista)
                ymin, ymax = min(y_lista), max(y_lista)
                bbox = (xmin, xmax, ymin, ymax)

                if dibujar:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return lista, bbox


    def contar_manos(self):
        if self.resultados and self.resultados.multi_hand_landmarks:
            return len(self.resultados.multi_hand_landmarks)
        return 0
