import os
import math
import time
import cvzone
from ultralytics import YOLO
import cv2
from tracker import Tracker
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = YOLO("../YOLOv8-weights/best.pt")

# model = YOLO("../YOLOv8-weights/yolov8l.pt")
model = YOLO("../YOLOv8-weights/yolov8n.pt")

# Move the model to the chosen device (GPU if available)
if device.type == "cuda":
    model.to(device)

# Capture using a video
# video_path = os.path.join('..', 'images', 'people.mp4')
# cap = cv2.VideoCapture(video_path)


# Capture using webcam (real-time)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

ret, frame = cap.read()
tracker = Tracker()

# generate random colors
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(10)]

# class_names = [
#     "Alarousa black tea",
#     "Chipsy Shatta and Lemon",
#     "Coca-Cola can",
#     "Doritos",
#     "HOHOs mix",
#     "Indomie",
#     "Molto",
#     "Sprite can",
#     "Windows",
#     "chipsy Marinated Cheese",
#     "jaguar stix cheeseburger",
# ]
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

while True:

    new_frame_time = time.time()
    ret, frame = cap.read()
    results = model(frame, stream=True)

    for result in results:
        detections = []

        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
            # Confidence
            conf = math.ceil((conf * 100)) / 100
            # conf = round(conf, 2)
            detections.append([x1, y1, x2, y2, conf])

        # print(detections)
        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = track.track_id
            rec_color = (colors[track_id % len(colors)])

            # Put rectangle & text above it using cv2(openCV)
            cv2.rectangle(frame, (x1, y1), (x2, y2), rec_color, 3)
            cv2.putText(frame, f'{classNames[class_id]} {conf}', (max(0, x1), max(35, y1 - 12)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1.0,
                        color=(125, 246, 55),
                        thickness=2)

            # Put rectangle using cvzone
            # w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(frame, (x1, y1, w, h))
            # cvzone.putTextRect(frame, f'{classNames[class_id]} {conf}', (max(0, x1 + 8), max(35, y1 - 13)), scale=1,
            #                    thickness=1)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print(f"FPS: {fps}")

    # Display FPS with device information
    device_text = "CPU" if device.type == "cpu" else "GPU"
    cv2.putText(frame, f"FPS: {fps:.2f} ({device_text})", (10, 30),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=1)

    cv2.imshow("frame", frame)
    # cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
