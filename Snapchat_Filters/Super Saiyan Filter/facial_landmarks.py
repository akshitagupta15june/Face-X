import cv2
import dlib
import numpy as np

# Open webcam video capturer
cap = cv2.VideoCapture(0)

# Overlay Configurations
color_green = (0,255,0)
line_width = 3

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces from the grayscale image
    faces = detector(gray)
    for face in faces:
        
        # Draw bounding boxes around faces
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_green, 3)

        # Plot the facial landmark points on the frame
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, color_green, -1)
 
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

'''
Run: python3 facial_landmarks.py
'''
