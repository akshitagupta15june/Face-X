import numpy as np
import cv2
import os
from imutils import resize
import pickle


def calc_Embeddings(all_files,names):

	detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")
	embedder = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")

	knownNames = []
	knownEmbeddings = []
	total = 0
	for dir in all_files:
		name = names[total]
		for file in dir:
			file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
			image = cv2.imdecode(file_bytes, 1)
			image = resize(image, width=600)
			(h, w) = image.shape[:2]

			imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
			detector.setInput(imageBlob)
			detections = detector.forward()

			if len(detections) > 0:
				i = np.argmax(detections[0, 0, :, 2])
				confidence = detections[0, 0, i, 2]

				if confidence > 0.5:
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					face = image[startY:endY, startX:endX]
					(fH, fW) = face.shape[:2]
					if fW < 20 or fH < 20:
						continue

					faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
					embedder.setInput(faceBlob)
					vec = embedder.forward()
					knownNames.append(name)
					knownEmbeddings.append(vec.flatten())
		total += 1

	print(knownEmbeddings)
	print(len(knownEmbeddings))
	print(knownNames)
	print(len(knownNames))
	with open("unknownEmbeddings.pkl","rb") as fp:
		l = pickle.load(fp)
	with open("unknownNames.pkl","rb") as fp:
		n = pickle.load(fp)
	for i in l:
		knownEmbeddings.append(i)
	knownNames = knownNames + n 
	print(knownEmbeddings)
	print(len(knownEmbeddings))
	print(knownNames)
	print(len(knownNames))
	return knownEmbeddings,knownNames


