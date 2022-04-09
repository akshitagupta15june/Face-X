import numpy as np
import cv2
import os
from imutils import resize
import streamlit as st

def detect(video,model,le):
	detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")
	embedder = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")
	cap = cv2.VideoCapture(video)
	stframe = st.empty()

	while True:
		ret,frame = cap.read()

		if ret:
			frame = resize(frame, width=600)
			(h, w) = frame.shape[:2]
			imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
			detector.setInput(imageBlob)
			detections = detector.forward()

			for i in range(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > 0.5:
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					face = frame[startY:endY, startX:endX]
					(fH, fW) = face.shape[:2]
					if fW < 20 or fH < 20:
						continue

					faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
					embedder.setInput(faceBlob)
					vec = embedder.forward()
					preds = model.predict_proba(vec)[0]
					j = np.argmax(preds)
					proba = preds[j]
					name = le.classes_[j]
					text = "{}: {:.2f}%".format(name, proba * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
					cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
					# cv2.imshow("frame",frame)
					stframe.image(frame)


		# cv2.waitKey(0)
