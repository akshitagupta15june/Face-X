import cv2
import numpy as np


def check(image_path):
	# Reading the image
	img = cv2.imread(image_path)

	# Parameters for Bilateral Filter

	# Diameter of the pixel neighborhood — the larger this diameter is, 
	# the more pixels will be included in the blurring computation.
	diameter = 11

	# SigmaColor is the number of colors in the neighborhood 
	# that will be considered when computing the blur.
	sigmaColor = 61

	# The value of SigmaSpace indicates pixels farther out 
	# from the central pixel diameter will influence the blurring calculation.
	sigmaSpace = 39

	# Applying Bilateral Filter
	blurred = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)

	# Converting original image to HSV
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Converting blurred image to hsv
	blurred_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# Subtracting blurred from original and storing value
	isNotACartoonIndex = np.mean( img_hsv - blurred_hsv )

	if isNotACartoonIndex >= 55:
		print("Human Face")
	else:
		print("Cartoon Face")


check("/Assets/cartoon.jpeg")
check("/Assets/image.webp")