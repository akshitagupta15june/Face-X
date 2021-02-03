import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
nose_image = cv2.imread("pig_nose.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # nose coordinates
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)

        nose_width = int(hypot(left_nose[0] - right_nose[0],
                               left_nose[1] - right_nose[1]))

        nose_height = int(nose_width * 1)

        # New nose position
        top_left = (int(center_nose[0] - nose_width/2),
                    int(center_nose[1] - nose_height/2))

        bottom_right = (int(center_nose[0] + nose_width/2),
                        int(center_nose[1] + nose_height/2))

        # Addign the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(
            nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                          top_left[0]: top_left[0] + nose_width]

        nose_area_no_nose = cv2.bitwise_and(
            nose_area, nose_area, mask=nose_mask)

        final_nose = cv2.add(nose_area_no_nose, nose_pig)

        frame[top_left[1]: top_left[1] + nose_height,
              top_left[0]: top_left[0] + nose_width] = final_nose

        # cv2.imshow("Nose Area", nose_area)
        # cv2.imshow("Nose Pig",nose_pig)
        # cv2.imshow("Nose Area no Nose", nose_area_no_nose)
        #cv2.imshow("Final Nose", final_nose)

        # print(nose_width)
        # print(nose_height)

        #cv2.circle(frame, top_nose, 3, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Pig Nose", nose_pig)

    key = cv2.waitKey(1)

    if key == 27:
        break
