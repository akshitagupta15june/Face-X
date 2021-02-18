#Importing required libraries
import cv2
import numpy as np

#Reading image 
img = cv2.imread("face.jpg") 
cv2.imshow("Original image",img)

#Converting to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Converted to RGB",img)

#Detecting edges of the input image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
cv2.imshow("GrayScale",gray)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
cv2.imshow("Dectected edges",edges)

#Cartoonifying the image
color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)

cv2.imshow("After Cartoonification",cartoon)
cv2.imwrite("Cartoon.jpg",cartoon)
