import cv2
import numpy as np
# Reading the Image
image = cv2.imread("Images/face.jpg")
# Finding the Edges of Image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
# Making a Cartoon of the image
color = cv2.bilateralFilter(image, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)
#Visualize the cartoon image
cv2.imshow("Images/face.jpg",image)
cv2.imshow("Cartoon Image", cartoon)
cv2.imwrite("Carttonified.jpg",cartoon)
cv2.waitKey(0) # "0" is Used to close the image window
cv2.destroyAllWindows()
