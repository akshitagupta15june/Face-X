import cv2
import dlib 
import numpy as np
from math import hypot

camera_video = cv2.VideoCapture(0)
scar = cv2.imread("HP_scar.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    
    _, frame = camera_video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Applying scar filter on all faces detected by the camera
    for face in faces:
        
        # Identifying landmarks 19,20,21 present on the right eyebrow
        landmarks = predictor(gray, face)
        left = (landmarks.part(19).x, landmarks.part(19).y)
        right = (landmarks.part(21).x, landmarks.part(21).y)
        middle = (landmarks.part(20).x, landmarks.part(20).y)
        
        # Resizing the scar while maintaining the aspect ratio 
        scar_width = int(hypot(left[0] - right[0], left[1] - right[1]))
        scar_height = int(scar_width * scar.shape[0] / scar.shape[1]) 
        resized_scar = cv2.resize(scar, (scar_width, scar_height)) 
        
        # Identifying scar position on face
        top_left = (int(middle[0] - scar_width/2), int(middle[1]- scar_height - scar_width / 2))
        face_area = frame[top_left[1] : top_left[1] + scar_height, top_left[0] : top_left[0] + scar_width]
         
        # Graying and thresholding the scar
        gray_scar = cv2.cvtColor(resized_scar, cv2.COLOR_BGR2GRAY)
        _, thresh_scar = cv2.threshold(gray_scar, 25, 255, cv2.THRESH_BINARY_INV)
        
        # Adding the scar on the face 
        face_area_no_face = cv2.bitwise_and(face_area, face_area, mask = thresh_scar)
        final_mask = cv2.add(face_area_no_face, resized_scar)
        frame[top_left[1] : top_left[1] + scar_height, top_left[0] : top_left[0] + scar_width] = final_mask

    cv2.imshow("Frame", frame)
   
    # Breaking the loop if 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF    
    if(key == 27):
        break

# Releasing the VideoCapture Object and closing the windows.                  
camera_video.release()
cv2.destroyAllWindows()