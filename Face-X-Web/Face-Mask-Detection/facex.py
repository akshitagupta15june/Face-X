import imutils
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


class InvalidFile(Exception):
	pass
                                                                                                                       
def cartoonify(img_path, method='opencv'):

	if(os.path.isfile(img_path)):

		try:

			if method=='opencv':
				# Reading the Image
				image = cv2.imread(img_path)
				# Finding the Edges of Image
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				blur = cv2.medianBlur(gray, 5)
				edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
				# Making a Cartoon of the image
				color = cv2.bilateralFilter(image, 9, 250, 250)
				cartoon = cv2.bitwise_and(color, color, mask=edges)

				return cartoon
		except:

			raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")

	else:

		raise InvalidFile("Invalid File!")



def face_detection(img_path, method='opencv'):	

	# Code by : Srimoni Dutta
	# Link : https://github.com/akshitagupta15june/Face-X/tree/master/Face-Detection/Face%20Detection%20using%20Haar%20Cascade

	if(os.path.isfile(img_path)):

		try:

			if method=='opencv':
				face_cascade=cv2.CascadeClassifier(os.path.join(os.getcwd(),'haarcascade_frontalface_default.xml'))
				img=cv2.imread(img_path)
				gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				faces=face_cascade.detectMultiScale(gray,1.1,4)
				for (x,y,w,h) in faces:
					cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

				return img
		except:

			raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")

	else:

		raise InvalidFile("Invalid File!")				

def blur_bg(img_path, method='opencv'):

	# Code by : Anas-Issa
	# Link : https://github.com/akshitagupta15june/Face-X/tree/master/Blurring%20image%20across%20face

	if(os.path.isfile(img_path)):

		try:

			if method=='opencv':

				detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
				img=cv2.imread(img_path)

				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				faces = detector.detectMultiScale(gray, 1.3, 5)

				for (x, y, w, h) in faces:
						face = img[y:y + h, x:x + w]
						frame = cv2.blur(img, ksize = (10, 10))
						frame[y:y + h, x:x + w] = face

				return img
		except:

			raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")

	else:

		raise InvalidFile("Invalid File!")							

def ghost_img(img_path, method='opencv'):

	# Code by : A-kriti
	# Link : https://github.com/akshitagupta15june/Face-X/tree/master/Awesome-face-operations/Ghost%20Image

	if(os.path.isfile(img_path)):

		try:	

			if method=='opencv':
				# take path of the image as input
				img_path = img_path   
				img = cv2.imread(img_path)

				image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

				img_small = cv2.pyrDown(image)
				num_iter = 5
				for _ in range(num_iter):
					img_small= cv2.bilateralFilter(img_small, d=9, sigmaColor=9, sigmaSpace=7)
				img_rgb = cv2.pyrUp(img_small)

				img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
				img_blur = cv2.medianBlur(img_gray, 7)
				img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

				img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

				array = cv2.bitwise_xor(image, img_edge)      #used bitwise xor method 
				plt.figure(figsize=(10,10))
				plt.imshow(array)
				plt.axis('off')
				filename = os.path.basename(img_path)

				return array  #final filtered photo

		except:

			raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")

	else:

		raise InvalidFile("Invalid File!")						



def mosaic (img_path, x, y, w, h, neighbor=9):

	# Code by : Sudip Ghosh
	# Link : https://github.com/AdityaNikhil/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic.py

	if(os.path.isfile(img_path)):

		try:

			frame = cv2.imread(img_path, 1)	
			fh, fw=frame.shape [0], frame.shape [1]
			if (y + h>fh) or (x + w>fw):
				return
			for i in range (0, h-neighbor, neighbor):#keypoint 0 minus neightbour to prevent overflow
				for j in range (0, w-neighbor, neighbor):
					rect=[j + x, i + y, neighbor, neighbor]
					color=frame [i + y] [j + x] .tolist () #key point 1 tolist
					left_up=(rect [0], rect [1])
					right_down=(rect [0] + neighbor-1, rect [1] + neighbor-1) #keypoint 2 minus one pixel
					cv2.rectangle (frame, left_up, right_down, color, -1)

			return frame
		except:

			raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")

	else:

		raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")

def sketch(img_path, method='opencv'):
	
	# Code by : iaditichine
	# Link : https://github.com/akshitagupta15june/Face-X/blob/master/Awesome-face-operations/Pencil%20Sketch/pencil_sketch_code.py		

	if(os.path.isfile(img_path)):

		try:

			img=cv2.imread(img_path)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img_invert = cv2.bitwise_not(img_gray)
			img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
			
			return cv2.divide(img_gray, 255 - img_smoothing, scale=256)
		except:

			raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")

	else:

		raise InvalidFile("Only image files are supported.(.png, .jpg, .jpeg)")	


def detect_and_predict_mask(frame, faceNet, maskNet):

	# (Partly taken)Code by : PyImageSearch
	# Link : https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

	# grab the dimensions of the frame and then construct a blob from it
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# ensure the bounding boxes fall within the dimensions of
			# the frame
			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def face_mask(image):
	from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
	from tensorflow.keras.preprocessing.image import img_to_array
	from tensorflow.keras.models import load_model
	from imutils.video import VideoStream	
	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = "models/deploy.prototxt"
	weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
	model = "models/mask_detector.model"
	net = cv2.dnn.readNet(prototxtPath, weightsPath)
	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(model)	

	if(os.path.isfile(image)):
		#############################################

		## IMAGE MASK DETECTION

		#############################################		
		if(image.endswith('jpg') or image.endswith('png')):
			try:
				# load the input image from disk, clone it, and grab the image spatial
				# dimensions
				image = cv2.imread(image)
				orig = image.copy()
				(h, w) = image.shape[:2]
				# construct a blob from the image
				blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
					(104.0, 177.0, 123.0))
				# pass the blob through the network and obtain the face detections
				print("[INFO] computing face detections...")
				net.setInput(blob)
				detections = net.forward()

				# loop over the detections
				for i in range(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated with
					# the detection
					confidence = detections[0, 0, i, 2]
					# filter out weak detections by ensuring the confidence is
					# greater than the minimum confidence
					if confidence > 0.5:
						# compute the (x, y)-coordinates of the bounding box for
						# the object
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						# ensure the bounding boxes fall within the dimensions of
						# the frame
						# extract the face ROI, convert it from BGR to RGB channel
						# ordering, resize it to 224x224, and preprocess it
						face = image[startY:endY, startX:endX]
						face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
						face = cv2.resize(face, (224, 224))
						face = img_to_array(face)
						face = preprocess_input(face)
						face = np.expand_dims(face, axis=0)
						# pass the face through the model to determine if the face
						# has a mask or not
						(mask, withoutMask) = model.predict(face)[0]

				# determine the class label and color we'll use to draw
						# the bounding box and text
						label = "Mask" if mask > withoutMask else "No Mask"
						color = (255, 255, 0) if label == "Mask" else (0, 255, 255)
						# include the probability in the label
						label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
						# display the label and bounding box rectangle on the output
						# frame
						cv2.putText(image, label, (startX, startY - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 3)
						cv2.rectangle(image, (startX, startY), (endX, endY), color, 10)

					return image

			except:

				raise InvalidFile("Files of following format are only supported.(.png, .jpg, .jpeg, .mp4)")


		#############################################

		## VIDEO MASK DETECTION

		#############################################

		elif(image.endswith('mp4')):
			print("[INFO] starting video stream...")
			cap = cv2.VideoCapture(image)

			# loop over the frames from the video stream
			while cap.isOpened():
				# grab the frame from the threaded video stream and resize it
				# to have a maximum width of 400 pixels
				ret,frame = cap.read()
				frame = imutils.resize(frame, width=400)
				# detect faces in the frame and determine if they are wearing a
				# face mask or not
				(locs, preds) = detect_and_predict_mask(frame, net, model)

				# loop over the detected face locations and their corresponding
				# locations
				for (box, pred) in zip(locs, preds):
					# unpack the bounding box and predictions
					(startX, startY, endX, endY) = box
					(mask, withoutMask) = pred
					# determine the class label and color we'll use to draw
					# the bounding box and text
					label = "Mask" if mask > withoutMask else "No Mask"
					color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
					# include the probability in the label
					label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
					# display the label and bounding box rectangle on the output
					# frame
					cv2.putText(frame, label, (startX, startY - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

				# show the output frame
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF
				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break
			# do a bit of cleanup
			cv2.destroyAllWindows()
			cap.release()



	else:

		raise InvalidFile("Files of following format are only supported.(.png, .jpg, .jpeg, .mp4)")


