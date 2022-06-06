import cv2
import dlib 
import numpy as np
from math import hypot

camera_video = cv2.VideoCapture(0)
bandana = cv2.imread("bandana.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    
    _, frame = camera_video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Applying bandana filter on all faces detected by the camera
    for face in faces:
        
        # Identifying facial landmarks 0, 16, 29 
        landmarks = predictor(gray, face)
        left = (landmarks.part(0).x, landmarks.part(0).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)
        middle = (landmarks.part(29).x, landmarks.part(29).y)
        
        # Resizing the bandana
        bandana_width = int(1.4*hypot(left[0] - right[0], left[1] - right[1]))
        bandana_height = int(bandana_width * bandana.shape[0] / bandana.shape[1]) 
        resized_bandana = cv2.resize(bandana, (bandana_width, bandana_height)) 
        
        # Identifying bandana position on face
        top_left = (int(middle[0] - bandana_width / 2), int(middle[1]- bandana_height))
        face_area = frame[top_left[1] : top_left[1] + bandana_height, top_left[0] : top_left[0] + bandana_width]
         
        # Graying and thresholding the bandana
        gray_bandana = cv2.cvtColor(resized_bandana, cv2.COLOR_BGR2GRAY)
        _, thresh_bandana = cv2.threshold(gray_bandana, 25, 255, cv2.THRESH_BINARY_INV)
        
        # Adding the bandana on the face 
        face_area_no_face = cv2.bitwise_and(face_area, face_area, mask = thresh_bandana)
        final_mask = cv2.add(face_area_no_face, resized_bandana)
        frame[top_left[1] : top_left[1] + bandana_height, top_left[0] : top_left[0] + bandana_width] = final_mask

    cv2.imshow("Frame", frame)
   
    # Breaking the loop if 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF    
    if(key == 27):
        break

# Releasing the VideoCapture Object and closing the windows.                  
camera_video.release()
cv2.destroyAllWindows()