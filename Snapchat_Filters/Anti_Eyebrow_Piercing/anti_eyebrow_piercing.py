import cv2
import dlib 
import numpy as np
from math import hypot

camera_video = cv2.VideoCapture(0)
stud=cv2.imread("stud.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    
    _, frame = camera_video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Applying stud filter on all faces detected by the camera
    for face in faces:

        # Identifying facial landmarks 46, 16, 15 
        landmarks = predictor(gray, face)
        left = (landmarks.part(45).x, landmarks.part(45).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)
        middle = (landmarks.part(15).x, landmarks.part(15).y)

        # Resizing the stud
        stud_width = int(1.6*hypot(left[0] - right[0], left[1] - right[1]))
        stud_height = int(1.6*stud_width * stud.shape[0] / stud.shape[1]) 
        resized_stud = cv2.resize(stud, (stud_width, stud_height)) 

        # Identifying stud position on face
        top_left = (int(middle[0] - stud_width), int(middle[1]- stud_height/2))
        face_area = frame[top_left[1] : top_left[1] + stud_height, top_left[0] : top_left[0] + stud_width]

        # Graying and thresholding the stud
        gray_stud = cv2.cvtColor(resized_stud, cv2.COLOR_BGR2GRAY)
        _, thresh_stud = cv2.threshold(gray_stud, 25, 255, cv2.THRESH_BINARY_INV)

        # Adding the stud on the face 
        face_area_no_face = cv2.bitwise_and(face_area, face_area, mask = thresh_stud)
        final_mask = cv2.add(face_area_no_face, resized_stud)
        frame[top_left[1] : top_left[1] + stud_height, top_left[0] : top_left[0] + stud_width] = final_mask

    cv2.imshow("Frame", frame)
   
    # Breaking the loop if 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF    
    if(key == 27):
        break

# Releasing the VideoCapture Object and closing the windows.                  
camera_video.release()
cv2.destroyAllWindows()