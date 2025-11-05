import cv2
import numpy as np
import math
from ultralytics import YOLO
from cvzone.ClassificationModule import Classifier

# -------------------------------
# Load YOLO hand detector
# -------------------------------
# You can replace 'yolov8n-pose.pt' with a custom hand-detection model
model_detector = YOLO('yolov8n-pose.pt')

# Load your trained gesture classifier
classifier = Classifier('model/model.h5', 'model/labels.txt')

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

# Image preprocessing parameters
imgSize = 400
offset = 20

while True:
    success, img = cap.read()
    if not success:
        print("Failed to access camera.")
        break

    # Run YOLO inference
    results = model_detector(img, verbose=False)

    # Process each detection
    for result in results:
        if result.boxes is None or len(result.boxes.xyxy) == 0:
            continue

        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            # Safety bounds for crop
            y1_crop = max(0, y1 - offset)
            y2_crop = min(img.shape[0], y2 + offset)
            x1_crop = max(0, x1 - offset)
            x2_crop = min(img.shape[1], x2 + offset)

            imgCrop = img[y1_crop:y2_crop, x1_crop:x2_crop]

            # Skip if crop is empty
            if imgCrop.size == 0:
                continue

            # Prepare white canvas
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            # Resize and center the image
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Classification (using the preprocessed image)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = classifier.labels[index]

            # Draw bounding box & label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(35, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Detection", img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
