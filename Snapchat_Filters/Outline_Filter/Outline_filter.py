import cv2
import argparse 

import numpy as np

# using ArguementParser to create arguement lines to obtain input image
arg_pars = argparse.ArgumentParser()
arg_pars.add_argument("-i", "--image", required=True, help="path to input image")

args = vars(arg_pars.parse_args())

# Loading and reading image with openCV
input_image = cv2.imread(args["image"])
# convert to grayscale and obtain the edges of the image with canny edge detection
grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
outline_img = cv2.Canny(cv2.bilateralFilter(grayscale, 7, 60, 60), 60, 120)

# providing value to position and shift the outline from the real image
move_img = 15
padd = np.pad(outline_img,((0,0),(0, move_img)), mode="constant")

# overlaying the real image with the edge layer outline
gray_outline = cv2.add(grayscale, padd[:,move_img:])

Final_out_img = np.stack((padd[:,move_img:],)*3, axis=-1)

original_outline = cv2.add(input_image, Final_out_img)

# Display output image
cv2.imshow("Outline filter", original_outline)
cv2.waitKey(0)