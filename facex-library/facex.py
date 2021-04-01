import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


class InvalidFile(Exception):
	pass

def cartoonify(img_path, method='opencv'):

	# Code by : Sagnik Mukherjee
	# Link : https://github.com/akshitagupta15june/Face-X/tree/master/Cartoonify%20Image

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
				face_cascade=cv2.CascadeClassifier('facex/Face_Detection_using_OpenCV/haarcascade_frontalface_default.xml')
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

				detector = cv2.CascadeClassifier("facex/Face_Detection_using_OpenCV/haarcascade_frontalface_default.xml")
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




				
