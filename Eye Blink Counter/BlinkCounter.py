import cv2
import cvzone

cap = cv2.VideoCapture('Video.mp4')

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    cv2.imshow("Image",img)
    cv2.waitKey(1)
