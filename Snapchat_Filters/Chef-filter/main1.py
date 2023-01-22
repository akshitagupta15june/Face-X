from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math

tiara = cv2.imread("assets/tiara.png")
cap = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        try:

            landmarks = predictor(gray, face)
            landmarks_np = face_utils.shape_to_np(landmarks)

            # Tiara HEAD
            bottom_left = (landmarks.part(0).x, landmarks.part(0).y)
            bottom_right = (landmarks.part(16).x, landmarks.part(16).y)
            fore_wd = int(math.hypot(
                bottom_left[0] - bottom_right[0], bottom_left[1] - bottom_right[1]))
            fore_adj = imutils.resize(tiara, width=fore_wd)
            fore_ht = fore_adj.shape[0]

            bottom_left = (int(bottom_left[0]-fore_wd//4),
                           int(bottom_left[1]-fore_ht*1.7))
            bottom_right = (int(bottom_right[0]+fore_wd//4),
                            int(bottom_right[1]-fore_ht*1.7))
            top_left = (bottom_left[0], bottom_left[1]-fore_ht)
            top_right = (bottom_right[0], bottom_right[1]-fore_ht)

            bottom_left = (landmarks.part(3).x, landmarks.part(3).y+tiara_ht)
            bottom_right = (landmarks.part(13).x,
                            landmarks.part(13).y+tiara_ht)

            tiara_wd = int(math.hypot(
                bottom_left[0] - bottom_right[0], bottom_left[1] - bottom_right[1]))
            tiara_adj = imutils.resize(tiara, width=tiara_wd)
            tiara_ht = tiara_adj.shape[0]
            tiara_gray = cv2.cvtColor(tiara_adj, cv2.COLOR_BGR2GRAY)
            _, tiara_mask = cv2.threshold(
                tiara_gray, 22, 255, cv2.THRESH_BINARY)

            tiara_area = frame[top_left[1]: top_left[1] +
                               tiara_ht, top_left[0]: top_left[0] + tiara_wd]
            tiara_area_no_tiara = cv2.subtract(
                tiara_area, cv2.cvtColor(tiara_mask, cv2.COLOR_GRAY2BGR))
            tiara_final = cv2.add(tiara_area_no_tiara, tiara_adj)
            frame[top_left[1]: top_left[1] + tiara_ht, top_left[0]
                : top_left[0] + tiara_wd] = tiara_final

        except Exception as err:
            print(err)
            continue

    cv2.imshow("Chef Hat", frame)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cv2.destroyAllWindows()