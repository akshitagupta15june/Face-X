# importing modules required
import cv2 as cv
from cvzone.FaceMeshModule import FaceMeshDetector
import pyautogui
import webbrowser

# get video feed from camera
cap = cv.VideoCapture(0)

# initialize face detector to detect only 1 face
detector = FaceMeshDetector(maxFaces=1)

# initializing variables and list related to blinking
blinkCounter = 0
waitCounter = 0
ratioList = []

# each key press followed by 0.1 second pause
pyautogui.PAUSE = 0.0

# open game online
webbrowser.open('https://www.towergame.app/')

while True:

    # read the image frame from camera
    ret, img = cap.read()

    # stop program if nothing detected from camera
    if not ret:
        break

    # find face
    img, faces = detector.findFaceMesh(img, draw=False)

    #only if face detected
    if faces:
        face = faces[0]  # as we are detecting only 1 face

        # points on face related to eyebrows
        eyeUp, eyeDown, eyeLeft, eyeRight = face[159], face[23], face[130], face[243]

        # calculating length between left point of eye and right point of eye, and between top and bottom part of eye
        lenVer, _ = detector.findDistance(eyeUp, eyeDown)
        lenHor, _ = detector.findDistance(eyeLeft, eyeRight)

        # calculating ratio between the 2 lengths we calculated
        ratio = int((lenVer/lenHor)*100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)

        # checking for blink
        if ratioAvg < 30 and waitCounter == 0:

            # click on screen
            pyautogui.click(button='left')

            waitCounter = 1

        # wait for 20 frames before detecting next blink
        if waitCounter != 0:
            waitCounter += 1
            if waitCounter > 20:
                waitCounter = 0

        # display face
        # cvzone.putTextRect(img, f'Blink Counter : {blinkCounter}', (50, 100))
        cv.imshow('Camera feed', img)

    # quit program using 'Q' key on keyboard
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# exit program
cap.release()
cv.destroyAllWindows()



