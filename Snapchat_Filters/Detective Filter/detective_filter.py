from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math

hat = cv2.imread("assets/det_hat.png")
cigar = cv2.imread("assets/detective_cigar.png")
cap = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while (True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces =  detector(gray)
	
	for face in faces:
		try:
		
			landmarks = predictor(gray, face)
			landmarks_np = face_utils.shape_to_np(landmarks)


			# Detective Hat		
			bottom_left = (landmarks.part(0).x, landmarks.part(0).y)
			bottom_right = (landmarks.part(16).x, landmarks.part(16).y)
			fore_wd = int(math.hypot(bottom_left[0] - bottom_right[0], bottom_left[1] - bottom_right[1]))
			fore_adj = imutils.resize(hat, width=fore_wd)
			fore_ht = fore_adj.shape[0]

			bottom_left = (int(bottom_left[0]-fore_wd//4), int(bottom_left[1]-fore_ht*1.7))
			bottom_right = (int(bottom_right[0]+fore_wd//4), int(bottom_right[1]-fore_ht*1.7))
			top_left = (bottom_left[0], bottom_left[1]-fore_ht)
			top_right = (bottom_right[0], bottom_right[1]-fore_ht)

			hat_wd = int(math.hypot(bottom_left[0] - bottom_right[0], bottom_left[1] - bottom_right[1]))
			hat_adj = imutils.resize(hat, width=hat_wd)
			hat_ht = hat_adj.shape[0]
			hat_gray = cv2.cvtColor(hat_adj, cv2.COLOR_BGR2GRAY)
			_, hat_mask = cv2.threshold(hat_gray, 22, 255, cv2.THRESH_BINARY)

			hat_area = frame[top_left[1]: top_left[1] + hat_ht, top_left[0]: top_left[0] + hat_wd]
			hat_area_no_hat = cv2.subtract(hat_area, cv2.cvtColor(hat_mask, cv2.COLOR_GRAY2BGR))
			hat_final = cv2.add(hat_area_no_hat, hat_adj)
			frame[top_left[1]: top_left[1] + hat_ht, top_left[0]: top_left[0] + hat_wd] = hat_final




			# Detective Cigar
			top_left = (landmarks.part(52).x, landmarks.part(52).y)
			top_right = (landmarks.part(13).x, landmarks.part(13).y)

			cigar_wd = int(math.hypot(top_left[0] - top_right[0], top_left[1] - top_right[1])*0.85)
			cigar_adj = imutils.resize(cigar, width=cigar_wd)
			cigar_ht = cigar_adj.shape[0]

			bottom_left = (landmarks.part(3).x, landmarks.part(3).y+cigar_ht)
			bottom_right = (landmarks.part(13).x, landmarks.part(13).y+cigar_ht)

			cigar_gray = cv2.cvtColor(cigar_adj, cv2.COLOR_BGR2GRAY)
			_, cigar_mask = cv2.threshold(cigar_gray, 25, 255, cv2.THRESH_BINARY_INV)
			cigar_area = frame[top_left[1]: top_left[1] + cigar_ht, top_left[0]: top_left[0] + cigar_wd]
			cigar_area_no_cigar = cv2.bitwise_and(cigar_area, cigar_area, mask=cigar_mask)
			cigar_final = cv2.add(cigar_area_no_cigar, cigar_adj)
			frame[top_left[1]: top_left[1] + cigar_ht, top_left[0]: top_left[0] + cigar_wd] = cigar_final
		
		except Exception as err:
			print(err)
			continue
		
	cv2.imshow("Detective Filter",frame)	
	q = cv2.waitKey(1)
	if q==ord("q"):
		break

cv2.destroyAllWindows()
