import cv2
import numpy as np
import facex as x
# Read input image
img = cv2.imread('3.jpg')

# Resize image
img = cv2.resize(img, (500, 600))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median blur to remove noise
gray = cv2.medianBlur(gray, 5)

# Detect edges using the Canny algorithm
edges = cv2.Canny(gray, 100, 200)

# Threshold the image to create a black and white cartoon version
ret, thresh = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

# Create a color version of the cartoon image
color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Apply bilateral filter to smooth out colors
color = cv2.bilateralFilter(color, 9, 250, 250)

# Create final cartoonified image by combining edges and colors
cartoon = cv2.bitwise_and(color, img)

# Display the cartoonified image
cv2.imshow('Cartoon', cartoon)

# Save the cartoonified image
cv2.imwrite('cartoon.jpg', cartoon)

# Wait for key press and exit
cv2.waitKey(0)
cv2.destroyAllWindows()
