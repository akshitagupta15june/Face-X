import cv2
import numpy as np
import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
clown_nose = cv2.imread("clowmn_nomse.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        # Defining the nose coordinates
        top_nose = (landmarks.part(28).x, landmarks.part(28).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(36).x, landmarks.part(36).y)

        nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 1.7)
        nose_height = int(nose_width * 0.77)

        # Defining the extremities of the area of imposition of the clown nose filter
        top_left = (int(center_nose[0] - nose_width / 2), int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))

        # Adding the new nose
        nose_clown = cv2.resize(clown_nose, (nose_width, nose_height))
        nose_clown_gray = cv2.cvtColor(nose_clown, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_clown_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose_clown = cv2.add(nose_area_no_nose, nose_clown)

        frame[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose_clown

        cv2.imshow("Clown Nose", nose_area)
        cv2.imshow("Clown Nose", nose_clown)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
