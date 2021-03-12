import cv2
import numpy as np

def cartoonify(img_path, method='opencv'):

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

def face_detection(img_path, method='opencv'):	

	if method=='opencv':
		face_cascade=cv2.CascadeClassifier('facex/Face_Detection_using_OpenCV/haarcascade_frontalface_default.xml')
		img=cv2.imread(img_path)
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces=face_cascade.detectMultiScale(gray,1.1,4)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

		return img

def blur_bg(img_path, method='opencv'):

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



				
