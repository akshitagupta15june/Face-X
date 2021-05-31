# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:21:09 2019

@author: s098a
"""

from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np


from keras.preprocessing import image

model = load_model('Final_Model_Face.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cropped_face = img[y:y+h, x:x+w]
    
    return cropped_face


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    
    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224,224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis = 0)
        pred = model.predict(img_array)
        print(pred)
        
        #name = "None Matching"
        
        if (pred[0][0]>0.4):
            name = "Ash"
        if (pred[0][1]>0.4):
            name = "Malav"
        if (pred[0][2]>0.4):
            name = "Nani"
        
        cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No Face Found", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
