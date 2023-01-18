import numpy as np
import cv2
import pyautogui as pagui
import time as time

def clicker(key):
    pagui.keyDown(key)
    time.sleep(1)
    pagui.keyUp(key)
    
def detectWink(frame, location, ROI, cascade):
    eyes = cascade.detectMultiScale(
        ROI, 1.15, 3, 0|cv2.CASCADE_SCALE_IMAGE, (10, 20)) 
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    if len(eyes)==1:
        clicker('space')
    return len(eyes) == 1    # number of eyes is one

def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(18, 18))
    gray_frame = clahe.apply(gray_frame)
    gray_frame = cv2.bilateralFilter(gray_frame, 5, 15, 15)

    # # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    # gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.05 # range is from 1 to ..
    minNeighbors = 2   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0(faster) or 0|cv2.CASCADE_SCALE_IMAGE(more accurate)
    minSize = (30,30) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)

    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y+h, x:x+w]
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_eye.xml')
    runonVideo(face_cascade, eye_cascade)
