#Firstly imported the required libraries
import numpy as np
import cv2
import os.path 
from matplotlib import pyplot as plt

# take path of the image as input
image_path = input("Enter the path here:")  #example -> C:\Users\xyz\OneDrive\Desktop\project\image.jpg  
img = cv2.imread(image_path)

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_small = cv2.pyrDown(image)
num_iter = 55
for _ in range(num_iter):
    image_small= cv2.bilateralFilter(image_small, d=10, sigmaColor=5, sigmaSpace=7)
image_rgb = cv2.pyrUp(image_small)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
image_blur = cv2.medianBlur(image_gray, 13)

image_edge = cv2.cvtColor(image_blur, cv2.COLOR_GRAY2RGB)

final_result = cv2.bitwise_and(image_edge, image)    #used bitwise and method between the image_edge and image

plt.figure(figsize= (10,10))
plt.imshow(final_result)
plt.axis('off')
filename = os.path.basename(image_path)
#print(filename)
plt.savefig("(Oil Paint Filtered)"+filename)  #saved file name as (Filtered)image_name.jpg

plt.show()  #final oil paint filtered photo
