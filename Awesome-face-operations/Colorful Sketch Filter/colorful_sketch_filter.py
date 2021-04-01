from matplotlib import pyplot as plt
import numpy as np
import cv2
import os.path  #imported the required libraries

# take path of the image as input
image_path = input("Enter the path here:")  #example -> C:\Users\xyz\OneDrive\Desktop\project\image.jpg  
img = cv2.imread(image_path)

#plt.imshow(img)
#plt.show()
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.figure(figsize= (10,10))
#plt.imshow(image)
#plt.show()
image_small = cv2.pyrDown(image)
num_iter = 5
for _ in range(num_iter):
    image_small= cv2.bilateralFilter(image_small, d=9, sigmaColor=9, sigmaSpace=7)
image_rgb = cv2.pyrUp(image_small)
#plt.imshow(image_rgb)
#plt.show()


image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
image_blur = cv2.medianBlur(image_gray, 7)
image_edge = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0.2)

#plt.imshow(image_edge)
#plt.show()
image_edge = cv2.cvtColor(image_edge, cv2.COLOR_GRAY2RGB)
#plt.imshow(image_edge)
#plt.show()

result = cv2.bitwise_or(image_edge, image)    #used bitwise or method between the image_edge and image

plt.figure(figsize= (10,10))
plt.imshow(result)
plt.axis('off')
filename = os.path.basename(image_path)
plt.savefig("(Colorful Sketch Filtered)"+filename)  #saved file name as (Filtered)image_name.jpg

plt.show()  #final colorful sketch filtered photo
