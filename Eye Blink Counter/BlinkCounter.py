import cv2
import cvzone

cap = cv2.VideoCapture('Video.mp4')

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    cv2.imshow("Image",img)
    cv2.waitKey(1)
