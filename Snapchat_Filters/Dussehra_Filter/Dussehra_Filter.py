import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 680)
cap.set(10, 170)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

crown = cv2.imread('Crown.jpg')

lower = np.array([0, 31, 0])
upper = np.array([48, 255, 255])
flag = 0



while True:

    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    bg = Image.fromarray(img)

    for (x, y, w, h) in faces:
        flag = 1
        scale = w/(crown.shape[1])

        # Preparing coordinates
        x1 = x
        x2 = x+w
        hy = int((crown.shape[0])*scale)
        y1 = max(y-hy, 0)
        y2 = y1+hy

        crown2 = cv2.resize(crown, (w, hy), interpolation=cv2.INTER_AREA)

        hsv_img = cv2.cvtColor(crown2, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower, upper)
        mask_inv = cv2.bitwise_not(mask)

        roi = img[y1:y2, x1:x2]

        # Preparing background and foreground -- blacking-out the area of crown in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(crown2, crown2, mask=mask)

        # Putting crown in ROI and modifying the main image
        dst = cv2.add(img1_bg, img2_fg)
        img[y1:y2, x1:x2] = dst

        w2 = min(w, x2-x1)
        if x2 < img.shape[1]-w2:
            img[y1:y2 + h, x2:x2 + w2] = img[y1:y2 + h, x1:x1 + w2]

        if x1 > w2:
            img[y1:y2 + h, x1 - w2:x1] = img[y1:y2 + h, x1:x1 + w2]

        if x2 < img.shape[1]-2*w2:
            img[y1:y2 + h, x2+w2:x2 + 2*w2] = img[y1:y2 + h, x1:x1 + w2]

        if x1 > 2*w2:
            img[y1:y2 + h, x1 - 2*w2:x1 - w2] = img[y1:y2 + h, x1:x1 + w2]

        if cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE) == 1:
            cv2.destroyAllWindows()

        cv2.imshow('Result', img)

        # Saving Image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('IAmRavana.jpg', img)

        elif cv2.waitKey(1) & 0xFF == ord('x'):
            break

    # If no face detected yet
    if flag == 0:
        cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
