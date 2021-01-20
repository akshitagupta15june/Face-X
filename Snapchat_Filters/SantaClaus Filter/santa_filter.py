from imutils import face_utils
import numpy as np
import cv2
import imutils
import dlib
import math

# loading img assets
santa_beard = cv2.imread("assets/santa_beard.png")
santa_hat = cv2.imread("assets/santa_hat.png")
# loading face recognition models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while (True):
	ret, frame = cap.read()	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces =  detector(gray)
	
	for face in faces:
		try:
		
			landmarks = predictor(gray, face)
			landmarks_np = face_utils.shape_to_np(landmarks)


			# Santa Beard
			top_left = (landmarks.part(3).x, landmarks.part(3).y)
			top_right = (landmarks.part(13).x, landmarks.part(13).y)
			
			beard_wd = int(math.hypot(top_left[0] - top_right[0], top_left[1] - top_right[1]))
			beard_adj = imutils.resize(santa_beard, width=beard_wd)
			beard_ht = beard_adj.shape[0]

			bottom_left = (landmarks.part(3).x, landmarks.part(3).y+beard_ht)
			bottom_right = (landmarks.part(13).x, landmarks.part(13).y+beard_ht)

			beard_gray = cv2.cvtColor(beard_adj, cv2.COLOR_BGR2GRAY)
			_, beard_mask = cv2.threshold(beard_gray, 25, 255, cv2.THRESH_BINARY_INV)
			beard_area = frame[top_left[1]: top_left[1] + beard_ht, top_left[0]: top_left[0] + beard_wd]
			beard_area_no_beard = cv2.bitwise_and(beard_area, beard_area, mask=beard_mask)
			beard_final = cv2.add(beard_area_no_beard, beard_adj)
			frame[top_left[1]: top_left[1] + beard_ht, top_left[0]: top_left[0] + beard_wd] = beard_final


			# Santa Cap
			bottom_left = (landmarks.part(0).x, landmarks.part(0).y)
			bottom_right = (landmarks.part(16).x, landmarks.part(16).y)

			hat_wd = int(math.hypot(bottom_left[0] - bottom_right[0], bottom_left[1] - bottom_right[1]))
			hat_adj = imutils.resize(santa_hat, width=hat_wd)
			hat_ht = hat_adj.shape[0]

			bottom_left = (bottom_left[0], bottom_left[1]-hat_ht//2)
			bottom_right = (bottom_right[0], bottom_right[1]-hat_ht//2)
			top_left = (bottom_left[0], bottom_left[1]-hat_ht)
			top_right = (bottom_right[0], bottom_right[1]-hat_ht)

			hat_gray = cv2.cvtColor(hat_adj, cv2.COLOR_BGR2GRAY)
			_, hat_mask = cv2.threshold(hat_gray, 30, 255, cv2.THRESH_BINARY_INV)
			hat_area = frame[top_left[1]: top_left[1] + hat_ht, top_left[0]: top_left[0] + hat_wd]
			hat_area_no_hat = cv2.bitwise_and(hat_area, hat_area, mask=hat_mask)
			hat_final = cv2.add(hat_area_no_hat, hat_adj)
			frame[top_left[1]: top_left[1] + hat_ht, top_left[0]: top_left[0] + hat_wd] = hat_final

		
		except Exception as err:
			print(err)
			continue
		
	cv2.imshow("Santa Filter",frame)	
	q = cv2.waitKey(1)
	if q==ord("q"):
		break

cv2.destroyAllWindows()
