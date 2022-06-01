import cv2
import dlib 
import numpy as np
from math import hypot

camera_video = cv2.VideoCapture(0)
headset = cv2.imread("vr_headset.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    
    _, frame = camera_video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Applying VR headset filter on all faces detected by the camera
    for face in faces:
        
        # Identifying facial landmarks 0, 16, 27 
        landmarks = predictor(gray, face)
        left = (landmarks.part(0).x, landmarks.part(0).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)
        middle = (landmarks.part(27).x, landmarks.part(27).y)
        
        # Resizing the VR headset
        headset_width = int(1.1*hypot(left[0] - right[0], left[1] - right[1]))
        headset_height = int(headset_width * headset.shape[0] / headset.shape[1]) 
        resized_headset = cv2.resize(headset, (headset_width, headset_height)) 
        
        # Identifying the VR headset position on face
        top_left = (int(middle[0] - headset_width / 2), int(middle[1]- headset_height / 2))
        face_area = frame[top_left[1] : top_left[1] + headset_height, top_left[0] : top_left[0] + headset_width]
         
        # Graying and thresholding the VR headset
        gray_headset = cv2.cvtColor(resized_headset, cv2.COLOR_BGR2GRAY)
        _, thresh_headset = cv2.threshold(gray_headset, 25, 255, cv2.THRESH_BINARY_INV)
        
        # Adding the VR headset on the face 
        face_area_no_face = cv2.bitwise_and(face_area, face_area, mask = thresh_headset)
        final_mask = cv2.add(face_area_no_face, resized_headset)
        frame[top_left[1] : top_left[1] + headset_height, top_left[0] : top_left[0] + headset_width] = final_mask

    cv2.imshow("Frame", frame)
   
    # Breaking the loop if 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF    
    if(key == 27):
        break

# Releasing the VideoCapture Object and closing the windows.                  
camera_video.release()
cv2.destroyAllWindows()