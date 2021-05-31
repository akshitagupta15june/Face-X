import cv2
from PIL import Image
import numpy as np



cap = cv2.VideoCapture(0)
ds_factor = 1
def show(image):
        mouth_cascade = cv2.CascadeClassifier('mouth.xml')

        if mouth_cascade.empty():
            raise IOError('Unable to load the mouth cascade classifier xml file')
        #open filter to apply
        fMask = Image.open('pic2.png')
      
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        bg = Image.fromarray(image)
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
        for (x,y,w,h) in mouth_rects:
            y = int(y - 0.15*h)
            
            cv2.rectangle(image,(x,y), (x+w,y+h), (0,255,0), 3)
            #paste it
            bg.paste(fMask, (x,y-(h//2)), mask=fMask)
            break
           
         #return as array
        return np.asarray(bg)

while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        cv2.imshow('Mouth Detector', show(frame))
        c = cv2.waitKey(1)
        if c == 27:
             break
print(cap)
cap.release()
cv2.destroyAllWindows()