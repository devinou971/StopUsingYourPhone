from ultralytics import YOLO
import cv2
from time import time, sleep
from datetime import datetime 
from pygame import mixer
from ultralytics.utils.plotting import Annotator

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(0)
mixer.init()


CONFIDENCE_THRESHOLD = 0.5
BACK_TO_WORK_SOUND = mixer.Sound("sounds/back_to_work.mp3")


t = time()
last_phone_detection = time() - 10

phone_frame = None
phone_confidence = 0.0

while True:
    phone_detected = False

    ret, frame = cap.read()
    results = model(frame)

    for box in results[0].boxes:
        cls = box.cls
        phone_detected = results[0].names[cls.item()] == "cell phone" and box.conf.item() >= CONFIDENCE_THRESHOLD
        if phone_detected:
            annotator = Annotator(frame)
            box_position = box.xyxy[0]
            annotator.box_label(box_position, "phone")
            phone_frame = annotator.result()
            phone_confidence = box.conf.item()
            break

    if phone_detected and time() - last_phone_detection > 10:
        cv2.imwrite(f"phoneFrame/{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}_({str(phone_confidence)[:5]}).jpg", frame)
        BACK_TO_WORK_SOUND.play()
        last_phone_detection = time()
    
    sleep(1)
