from matplotlib import pyplot as plt  #imported the required libraries
import numpy as np
import cv2
import os.path 


# take path of the image as input
image_path = input("Enter the path here:")  #example -> C:\Users\xyz\OneDrive\Desktop\project\image.jpg  
img = cv2.imread(image_path)

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_small = cv2.pyrDown(image)

image_rgb = cv2.pyrUp(image_small)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
image_blur = cv2.medianBlur(image_gray, 13)    #applied median blur on image_gray

image_edge = cv2.cvtColor(image_blur, cv2.COLOR_GRAY2RGB)

final_output = cv2.bitwise_or(image_edge, image)    #used bitwise or method between the image_edge and image

plt.figure(figsize= (15,15))
plt.imshow(final_output)
plt.axis('off')
filename = os.path.basename(image_path)
plt.savefig("(Glitter Filtered)"+filename)  #saved file name as (Filtered)image_name.jpg

plt.show()  #final glitter filtered photo
