import cv2
import numpy as np 

#path to classifiers
path = '/Users/mitchellkrieger/opt/anaconda3/envs/learn-env/share/opencv4/haarcascades/'

#get image classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

#read image
img = cv2.imread('people.jpg')

#convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#for each face
for (x,y,w,h) in faces:
    #draw rectangle around face
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #select face as region of interest 
    roi_gray = gray[y:y+h,x:x+h]
    roi_color = img[y:y+h,x:x+h]
    #within region of interest find eyes
    eyes = eye_cascade.detectMultiScale(roi_gray)
    #for each eye
    for (ex,ey,ew,eh) in eyes:
        #draw retangle around eye
        cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img) #shows image
cv2.waitKey(0) #waits until a key is pressed to progress
cv2.destroyAllWindows() #closes windows

