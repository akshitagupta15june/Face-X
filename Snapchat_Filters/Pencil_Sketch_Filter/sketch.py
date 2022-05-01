# Importing necessary libraries
import cv2


def pencil_sketch(image):
	# Converting the image from BGR to GRAY
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# Inverting the gray image
	inverted = cv2.bitwise_not(gray)

	# Blurring the inverted image
	blurred = cv2.GaussianBlur(inverted, (31,31),0)

	# Inverting the blurred image
	blurred_inverted = cv2.bitwise_not(blurred)

	# Performing bitwise division between the gray image and the inverted blurred image
	# to obtain a sketch like image
	sketch = cv2.divide(gray,blurred_inverted,scale=256.0)

	return sketch


# Reading the image
image = cv2.imread("girl.png")
sketched_image = pencil_sketch(image)

cv2.imshow("img",sketched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
