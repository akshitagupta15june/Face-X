import os
import random
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone

cap = cv2.VideoCapture(0)  # Change the webcam index if needed
cap.set(3, 1280)
cap.set(4, 720)

detector = FaceMeshDetector(maxFaces=1)
idList = [0, 17, 78, 292]

# import images
folderEatable = 'Catch-The-Eatable/eatable'  # Change the folder path accordingly
listEatable = os.listdir(folderEatable)
eatables = []
for object in listEatable:
    eatables.append(cv2.imread(f'{folderEatable}/{object}', cv2.IMREAD_UNCHANGED))

folderNonEatable = 'Catch-The-Eatable/noneatable'  # Change the folder path accordingly
listNonEatable = os.listdir(folderNonEatable)
nonEatables = []
for object in listNonEatable:
    nonEatables.append(cv2.imread(f'{folderNonEatable}/{object}', cv2.IMREAD_UNCHANGED))

currentObject = eatables[0]
pos = [300, 0]
speed = 5
count = 0
global isEatable
isEatable = True
gameOver = False


def resetObject():
    global isEatable, currentObject
    pos[0] = random.randint(100, 1180)
    pos[1] = 0
    randNo = random.randint(0, 2)  # change the ratio of eatables/ non-eatables
    if randNo == 0:
        currentObject = nonEatables[random.randint(0, 3)]
        isEatable = False
    else:
        currentObject = eatables[random.randint(0, 3)]
        isEatable = True


while True:
    success, img = cap.read()

    if not success:
        continue

    if gameOver is False:
        img, faces = detector.findFaceMesh(img, draw=False)

        if img is None:
            continue

        imgRGB = img

        imgRGB = cvzone.overlayPNG(imgRGB, currentObject, pos)
        pos[1] += speed

        if pos[1] > 520:
            resetObject()

        if faces:
            face = faces[0]
            up = face[idList[0]]
            down = face[idList[1]]

            for id in idList:
                cv2.circle(imgRGB, face[id], 5, (255, 0, 255), 5)
            cv2.line(imgRGB, up, down, (0, 255, 0), 3)
            cv2.line(imgRGB, face[idList[2]], face[idList[3]], (0, 255, 0), 3)

            upDown, _ = detector.findDistance(face[idList[0]], face[idList[1]])
            leftRight, _ = detector.findDistance(face[idList[2]], face[idList[3]])

            ## Distance of the Object
            cx, cy = (up[0] + down[0]) // 2, (up[1] + down[1]) // 2
            cv2.line(imgRGB, (cx, cy), (pos[0] + 50, pos[1] + 50), (0, 255, 0), 3)
            distMouthObject, _ = detector.findDistance((cx, cy), (pos[0] + 50, pos[1] + 50))
            print(distMouthObject)

            # Lip opened or closed
            ratio = int((upDown / leftRight) * 100)
            if ratio > 60:
                mouthStatus = "Open"
            else:
                mouthStatus = "Closed"
            cv2.putText(imgRGB, mouthStatus, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

            if distMouthObject < 100 and ratio > 60:
                if isEatable:
                    resetObject()
                    count += 1
                else:
                    gameOver = True
        cv2.putText(imgRGB, str(count), (1100, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
    else:
        cv2.putText(imgRGB, "Game Over", (300, 400), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 10)

    cv2.imshow("Image", imgRGB)
    key = cv2.waitKey(1)

    if key == ord('r'):
        resetObject()
        gameOver = False
        count = 0
        currentObject = eatables[0]
        isEatable = True
