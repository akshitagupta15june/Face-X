import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # face cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  # eye cascade

cap = cv2.VideoCapture(0)   # capturing the video
c=0

while True:
    ret,img = cap.read()  # reading the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversion into gray color
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # reading the face
    for (x,y,w,h) in faces:
        k=1
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  # drawing the rectangle around the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray) 
        for (ex,ey,ew,eh) in eyes:
            k=0
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)  # drawing the rectangle around the eye
        if k==1:
            out="You've blinked ",c," times"  # counting the blink
            c=c+1
            print(out)
        else:
            print("Not Blinking!")
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27: # press Escape key to stop the program.
        break
cap.release()
cv2.destroyAllWindows()