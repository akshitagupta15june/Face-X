import cv2
import dlib 
import numpy as np
from math import hypot

camera_video = cv2.VideoCapture(0)
gum = cv2.imread("bubblegum.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    
    _, frame = camera_video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Applying gum filter on all faces detected by the camera
    for face in faces:

        # Identifying facial landmarks 48, 54, 66 on the mouth 
        landmarks = predictor(gray, face)
        left = (landmarks.part(48).x, landmarks.part(48).y)
        right = (landmarks.part(54).x, landmarks.part(54).y)
        middle = (landmarks.part(66).x, landmarks.part(66).y)

        # Resizing the gum
        gum_width = int(1.1*hypot(left[0] - right[0], left[1] - right[1]))
        gum_height = int(gum_width * gum.shape[0] / gum.shape[1]) 
        resized_gum = cv2.resize(gum, (gum_width, gum_height)) 

        # Identifying gum position on face
        top_left = (int(middle[0] - gum_width / 2), int(middle[1]- gum_height/2))
        face_area = frame[top_left[1] : top_left[1] + gum_height, top_left[0] : top_left[0] + gum_width]

        # Graying and thresholding the gum
        gray_gum = cv2.cvtColor(resized_gum, cv2.COLOR_BGR2GRAY)
        _, thresh_gum = cv2.threshold(gray_gum, 25, 255, cv2.THRESH_BINARY_INV)

        # Adding the gum on the face 
        face_area_no_face = cv2.bitwise_and(face_area, face_area, mask = thresh_gum)
        final_mask = cv2.add(face_area_no_face, resized_gum)
        frame[top_left[1] : top_left[1] + gum_height, top_left[0] : top_left[0] + gum_width] = final_mask

    cv2.imshow("Frame", frame)
   
    # Breaking the loop if 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF    
    if(key == 27):
        break

# Releasing the VideoCapture Object and closing the windows.                  
camera_video.release()
cv2.destroyAllWindows()