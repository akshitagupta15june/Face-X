from matplotlib import pyplot as plt
import numpy as np
import cv2
import os.path

# take path of the image as input
img_path = input("Enter the path here:")  #example -> C:\Users\xyz\OneDrive\Desktop\project\image.jpg  
img = cv2.imread(img_path)



#plt.imshow(img)
#plt.show()
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.figure(figsize= (10,10))
#plt.imshow(image)
#plt.show()
img_small = cv2.pyrDown(image)
num_iter = 5
for _ in range(num_iter):
    img_small= cv2.bilateralFilter(img_small, d=9, sigmaColor=9, sigmaSpace=7)
img_rgb = cv2.pyrUp(img_small)
#plt.imshow(img_rgb)
#plt.show()
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)
img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
#plt.imshow(img_edge)
#plt.show()
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
#plt.imshow(img_edge)
#plt.show()
# img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
array = cv2.bitwise_xor(image, img_edge)      #used bitwise xor method 
plt.figure(figsize= (10,10))
plt.imshow(array)
plt.axis('off')
filename = os.path.basename(img_path)
plt.savefig("(Filtered)"+filename)  #saved file name as (Filtered)image_name.jpg

plt.show()  #final filtered photo
