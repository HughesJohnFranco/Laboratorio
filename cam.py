import cv2
import time
from mano import Mano
from ver_datos import mostrar_datos


def camara():
    cap = cv2.VideoCapture(1)
    detector = Mano()
    ptiempo = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 360))
        frame = detector.encontrar_manos(frame)

        num_manos = detector.contar_manos()
        ctiempo = time.time()
        fps = 1 / (ctiempo - ptiempo + 1e-6)
        ptiempo = ctiempo

        mostrar_datos(frame, num_manos, fps)

        cv2.imshow("PROTOTIPO_1_DETECTORMANOS", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
