import cv2

def mostrar_datos(frame, num_manos, fps):
    cv2.putText(frame, f'MANOS: {num_manos}', (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250), 1)
    cv2.putText(frame, f'FPS: {int(fps)}', (5, 40), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250), 1)
