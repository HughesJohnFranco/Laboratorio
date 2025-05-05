import os
import cv2
import numpy as np
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from zhelpers import create_folder, save_frames, mediapipe_detection, there_hand, draw_keypoints
from constantes import FONT, FONT_POS, FONT_SIZE

def capture_samples(path, margen_frame=1, min_cant_frames=10, delay_frames=3):
    create_folder(path)
    
    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            imagen = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margen_frame:
                    cv2.putText(imagen, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) >= min_cant_frames + margen_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: - (margen_frame + delay_frames)]
                    today = datetime.now().strftime('%y%m%d%H%M%S')
                    output_folder = os.path.join(path, f"sample_{today}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)

                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(imagen, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
            
            draw_keypoints(imagen, results)
            cv2.imshow('Captura', imagen)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    base_path = "frame_actions/mi_palabra"
    capture_samples(base_path)
