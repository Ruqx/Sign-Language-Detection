# 01_HandDetection_complete.py
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from pathlib import Path

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 400

folder = Path("data/") / "class_X"  # change "class_X" to the appropriate label folder
folder.mkdir(parents=True, exist_ok=True)
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Camera read failed")
        break

    hands, img = detector.findHands(img, flipType=False)  # returns list of hands with bbox
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # cvzone returns bbox this way
        # add offset and clip
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                # height is bigger -> fit height
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                # width is bigger or equal -> fit width
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # Save image
        save_path = folder / f"Image_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(save_path), imgWhite)
        counter += 1
        print("Saved:", save_path, " total:", counter)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
