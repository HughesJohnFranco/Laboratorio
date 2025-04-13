import cv2
import time
from mano import Mano
from config.ver_datos import mostrar_datos
from config.constantes import ANCHO_FRAME, ALTO_FRAME

class Camara:

    def __init__(self, fuente=1):
        self.cap = cv2.VideoCapture(fuente)
        self.detector = Mano()
        self.ptiempo = time.time()


    def esta_abierta(self):
        return self.cap.isOpened()


    def cerrar(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def leer_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.resize(frame, (ANCHO_FRAME, ALTO_FRAME))
        frame = self.detector.encontrar_manos(frame)

        ctiempo = time.time()
        fps = 1 / (ctiempo - self.ptiempo + 1e-6)
        self.ptiempo = ctiempo

        num_manos = self.detector.contar_manos()
        mostrar_datos(frame, num_manos, fps)

        return frame


    def ejecutar(self):
        while self.esta_abierta():
            frame = self.leer_frame()
            if frame is None:
                break

            cv2.imshow("PROTOTIPO_1_DETECTORMANOS", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cerrar()
