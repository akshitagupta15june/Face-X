#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import dlib


# In[6]:


cap=cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
glass_ori = cv2.imread('swirly.png',-1)

def transparentOverlay(src, overlay,pos=(0,0),scale=1):
    overlay = cv2.resize(overlay ,(0,0) ,fx= scale ,fy=scale)
    h,w, _ = overlay.shape #size of fg image
    rows,cols , _ =src.shape#size of bg image
    y ,x =pos[0],pos[1]
    
    for i in range(h):
        for j in range(w):
            if x+i > rows or y+j >=cols:
                continue
            alpha=float(overlay[i][j][3]/255) 
            src[x+i][y+j]= alpha+ overlay[i][j][:3]+[1-alpha]+src[x+i][y+j]
            
    return src        
            
        



while cap.isOpened():
     # Read the video stream
    result ,  frame = cap.read()
    
    if result:
         # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5,0, minSize=(120,120), maxSize=(350,350))
        for(x,y,w,h) in faces: # Apply the filter to each face detected
            if h>0 and w>0:
                glass_symin=int(y+1.5*h/5)
                glass_symax=int(y+2.5*h/5)
                sh_glass=glass_symax-glass_symin
               
                face_glass_ori=frame[glass_symin:glass_symax,x:x+w]
                glass = cv2.resize(glass_ori ,[ w ,sh_glass], interpolation = cv2.INTER_CUBIC)
                
                
                 # Overlay the filter onto the face region
                transparentOverlay(face_glass_ori,glass)
                
                
                
                
            
        
         # Display the resulting frame
        cv2.imshow("frame",frame)
        if cv2.waitKey(5)==ord('q'): # Exit the loop if 'q' is pressed
            break
cap.release()
cv2.destroyAllWindows() # Release the video capture and close all windows

