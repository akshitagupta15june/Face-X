import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


cap = cv2.VideoCapture('Video.mp4')
detector = FaceMeshDetector(maxFaces = 1)

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    img, faces = detector.findFaceMesh(img)

    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    cv2.imshow("Image",img)
    cv2.waitKey(1)
