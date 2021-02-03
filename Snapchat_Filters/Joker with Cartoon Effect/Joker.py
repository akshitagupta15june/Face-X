# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:18:34 2021

@author: dell
"""


import cv2
import numpy as np
import dlib
import math

cap = cv2.VideoCapture(0)

hat_image = cv2.imread("Hat.jpg")
lip_image = cv2.imread("Lips.jpg")
nose_image = cv2.imread("Nose.jpg")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

last_foreground = np.zeros((480, 640), dtype='uint8')


while (cap.isOpened()):
    _, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces =  detector(frame)
    
    for face in faces:
        
        #Hat
        landmarks = predictor(gray_frame, face)
        
        
        left_hat = (landmarks.part(17).x, landmarks.part(17).y)
        center_hat = (landmarks.part(21).x, landmarks.part(21).y)
        right_hat = (landmarks.part(26).x, landmarks.part(26).y)
        nose_top = (landmarks.part(27).x, landmarks.part(27).y)
        nose_bottom = (landmarks.part(30).x, landmarks.part(30).y)
        nose_height = math.sqrt((nose_top[0] - nose_bottom[0])**2 + 
                                (nose_top[1] - nose_bottom[1])**2)
        hat_width = int(math.hypot(left_hat[0] - right_hat[0], 
                           left_hat[1] - right_hat[1])*2)
        
        
        hat_height = int(hat_width*0.5)
        hat = cv2.resize(hat_image, (hat_width, hat_height))
        
        hat_gray = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)
        
        _, hat_mask = cv2.threshold(hat_gray, 25,255, cv2.THRESH_BINARY_INV)

        top_left = (int(center_hat[0]-hat_width/2),
                    int(center_hat[1] - hat_height/2 - nose_height*2))
        bottom_right  = (int(center_hat[0] +  hat_width/2),
                         int(center_hat[1] + hat_height*2 + nose_height))
        
        hat_area = frame[top_left[1]: top_left[1] + hat_height,
                          top_left[0]: top_left[0] + hat_width]
        
        hat_area_no_head = cv2.bitwise_and(hat_area, hat_area, mask =hat_mask)
        final_hat = cv2.add(hat_area_no_head, hat)
        
        frame[top_left[1]: top_left[1] + hat_height,
                          top_left[0]: top_left[0] + hat_width] = final_hat
        
        #Nose
        
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        
        
        nose_width = int(math.hypot(left_nose[0] - right_nose[0], 
                           left_nose[1] - right_nose[1]))
        
        
        nose_height = nose_width
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        
        _, nose_mask = cv2.threshold(nose_pig_gray, 25,255, cv2.THRESH_BINARY_INV)

        top_left = (int(center_nose[0]-nose_width/2),
                    int(center_nose[1] - nose_height/2))
        bottom_right  = (int(center_nose[0] +  nose_width/2),
                         int(center_nose[1] + nose_width/2))
        
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                          top_left[0]: top_left[0] + nose_width]
        
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask =nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)
        
        frame[top_left[1]: top_left[1] + nose_height,
                          top_left[0]: top_left[0] + nose_width] = final_nose
        
        
        #Lip
        
        
        left_lip = (landmarks.part(48).x, landmarks.part(48).y)
        center_lip = (landmarks.part(62).x, landmarks.part(62).y)
        right_lip = (landmarks.part(54).x, landmarks.part(54).y)
        
        
        lip_width = int(math.hypot(left_lip[0] - right_lip[0], 
                           left_lip[1] - right_lip[1])*1.5)
        
        
        lip_height = lip_width
        lip = cv2.resize(lip_image, (lip_width, lip_height))
        
        lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        
        _, lip_mask = cv2.threshold(lip_gray, 25,255, cv2.THRESH_BINARY_INV)

        top_left = (int(center_lip[0]-lip_width/2),
                    int(center_lip[1] - lip_height/2))
        bottom_right  = (int(center_lip[0] +  lip_width/2),
                         int(center_lip[1] + lip_width/2))
        
        lip_area = frame[top_left[1]: top_left[1] + lip_height,
                          top_left[0]: top_left[0] + lip_width]
        
        lip_area_no_lip = cv2.bitwise_and(lip_area, lip_area, mask =lip_mask)
        final_lip = cv2.add(lip_area_no_lip, lip)
        
        frame[top_left[1]: top_left[1] + lip_height,
                          top_left[0]: top_left[0] + lip_width] = final_lip
        
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        foreground = gray
    
        abs_diff = cv2.absdiff(foreground, last_foreground)
    
        last_foreground = foreground

        _, mask = cv2.threshold(abs_diff, 20, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=3)
        se = np.ones((85, 85), dtype='uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)

        frame_effect = cv2.stylization(frame, sigma_s=150, sigma_r=0.25)
        idx = (mask > 1)
        frame[idx] = frame_effect[idx]

        # cv2.imshow('WebCam (Mask)', mask)
    
        
        
            
    cv2.imshow("Frame", frame)
    #cv2.imshow("Pig Nose", nose_pig)
    key = cv2.waitKey(1);
    if key == 27:
        break
    