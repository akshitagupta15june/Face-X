import cv2

# Load the input image
input_image = cv2.imread('input_image.jpg')

# Convert the input image from BGR to HSV
hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

# Define the lower and upper HSV values for the red color that corresponds to the spiderman mask
lower_red = (0, 120, 70)
upper_red = (10, 255, 255)

# Create a mask for the spiderman mask by applying the inRange function to the input image using the lower and upper HSV values
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Apply the bitwise_and function to the input image and the mask to get the output image with the spiderman mask
output_image = cv2.bitwise_and(input_image, input_image, mask=mask)

# Show the output image
cv2.imshow('Spiderman Mask Filter', output_image)
cv2.waitKey(0)

# Save the output image
cv2.imwrite('output_image.jpg', output_image)

# Close all windows
cv2.destroyAllWindows()
