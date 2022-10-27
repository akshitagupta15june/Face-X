# Author Prijesh Devanand
# github id : https://github.com/KuberaPrijesh


#This file will convert the coloured image to black and white image, its working using OpenCv-Python in python.

import cv2

img=cv2.imread('dog.jpg')

bw_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("Original Image - input", img)

cv2.imshow("Black and White Image - output", bw_img)

cv2.waitKey(0)