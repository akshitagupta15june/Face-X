import cv2
import numpy as np 

path = '/Users/mitchellkrieger/opt/anaconda3/envs/learn-env/share/opencv4/haarcascades/'

#get facial classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

#read images
witch = cv2.imread('witch.png')

#get shape of witch
original_witch_h,original_witch_w,witch_channels = witch.shape

#get shape of img
#img_h,img_w,img_channels = img.shape

#convert to gray
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

#create mask and inverse mask of witch
ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#read video
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]

while True:   #continue to run until user breaks loop
    
    #read each frame of video and convert to gray
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #find faces in image using classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #for every face found:
    for (x,y,w,h) in faces:
        #retangle for testing purposes
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #witch size in relation to face by scaling
        witch_width = int(1.5 * face_w)
        witch_height = int(witch_width * original_witch_h / original_witch_w)
        
        #setting location of coordinates of witch
        witch_x1 = face_x2 - int(face_w/2) - int(witch_width/2)
        witch_x2 = witch_x1 + witch_width
        witch_y1 = face_y1 - int(face_h*1.25)
        witch_y2 = witch_y1 + witch_height 

        #check to see if out of frame
        if witch_x1 < 0:
            witch_x1 = 0
        if witch_y1 < 0:
            witch_y1 = 0
        if witch_x2 > img_w:
            witch_x2 = img_w
        if witch_y2 > img_h:
            witch_y2 = img_h

        #Account for any out of frame changes
        witch_width = witch_x2 - witch_x1
        witch_height = witch_y2 - witch_y1

        #resize witch to fit on face
        witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

        #take ROI for witch from background that is equal to size of witch image
        roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

        #original image in background (bg) where witch is not
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(witch,witch,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[witch_y1:witch_y2, witch_x1:witch_x2] = dst


        break
    #display image
    cv2.imshow('img',img) 

    #if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'): # 
        break;

cap.release() #turn off camera 
cv2.destroyAllWindows() #close all windows

