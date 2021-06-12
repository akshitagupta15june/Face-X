''' image pyramids are of two type that are:
1.Gaussian pyramid
2.Laplacian pyramid
below is the implemntation of gaussian pyramid using pyrDown()
fuction upto 5 levels'''
################### Laplacian Pyramid ###################
import cv2
import numpy as np
image = cv2.imread("ElonMusk.jpg")
image = cv2.resize(image,(256,256))
layers = image.copy()
gaussian_p = [layers]

for i in range(5):
    layers = cv2.pyrDown(layers) 
    gaussian_p.append(layers)
    #cv2.imshow(str(i), layers)

layers = gaussian_p[4]
cv2.imshow("this is upper level gaussian pyramid",layers)
lp = [layers]

for i in range(4, 0 ,-1):
    size = (gaussian_p[i-1].shape[1],gaussian_p[i-1].shape[0])
    gaussian_extend = cv2.pyrUp(gaussian_p[i], dstsize=size)
    laplacian = cv2.subtract(gaussian_p[i-1], gaussian_extend)
    cv2.imshow(str(i), laplacian)

cv2.imshow("original image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
