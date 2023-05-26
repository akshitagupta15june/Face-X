import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


cap = cv2.VideoCapture('Video.mp4')
# For detecting the blink using webcam comment the above line and uncomment the below line.
#cap = cv2.VideoCapture(2)  x
detector = FaceMeshDetector(maxFaces = 1)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img,face[id],3,(99,86,232),cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthVer, _ = detector.findDistance(leftUp,leftDown)
        lengthHor, _ = detector.findDistance(leftLeft,LeftRight)

        cv2.line(img, leftUp, leftDown, (251,238,106), 1)
        cv2.line(img, leftLeft, leftRight, (251,238,106), 1)
 
        print(int((lengthVer/lengthHor)*100))

    img = cv2.resize(img, (640, 360))
    cv2.imshow("Image",img)
    cv2.waitKey(1)
