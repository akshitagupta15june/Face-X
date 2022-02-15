import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
import cv2
import os

img_path='train'
image_list=os.listdir(img_path)
print(image_list[0:5])

#Looking at all the color space conversions OpenCV provides
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(len(flags))
print(flags[30:40])


#Loading an image
img_emma = cv2.imread(img_path+'/'+image_list[0])
plt.imshow(img_emma)
plt.show()

#By default opencv reads any assets in BGR format,
#converting from BGR to RGB color space

img_rgb = cv2.cvtColor(img_emma, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

#convering from BGR to GRAY color space
img_gray = cv2.cvtColor(img_emma, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray)
plt.show()

#converting from BGR to HSV color space
img_hsv = cv2.cvtColor(img_emma, cv2.COLOR_BGR2HSV)
plt.imshow(img_hsv)
plt.show()

#separting the channels of RBG imgae

#R channel
red = img_rgb.copy()
# set blue and green channels to 0
red[:, :, 1] = 0
red[:, :, 2] = 0
plt.imshow(red)
plt.show()


#G channel
green = img_rgb.copy()
green[:, :, 0] = 0
green[:, :, 2] = 0
plt.imshow(green)
plt.show()

#B channel
blue = img_rgb.copy()
blue[:, :, 0] = 0
blue[:, :, 1] = 0
plt.imshow(blue)
plt.show()



light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()

mask = cv2.inRange(img_hsv, light_orange, dark_orange)
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

#adding a second mask that looks for whites
light_white = (0, 0, 200)
dark_white = (145, 60, 255)

lw_square = np.full((10, 10, 3), light_white, dtype=np.uint8) / 255.0
dw_square = np.full((10, 10, 3), dark_white, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lw_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(dw_square))
plt.show()

mask_white = cv2.inRange(img_hsv, light_white, dark_white)
result_white = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_white)

plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()

#Adding mask together and plotting the result
final_mask = mask + mask_white

final_result = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()

blur = cv2.GaussianBlur(final_result, (7, 7), 0)
plt.imshow(blur)
plt.show()

#Applying segmenattion on list of assets
emma_images = []
for images in image_list[:5]:
   friend = cv2.cvtColor(cv2.imread(img_path +'/'+ images), cv2.COLOR_BGR2RGB)
   emma_images.append(friend)


def segment_image(image):
    

    # Convert the image into HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)

    # Apply the orange shade mask 
    mask = cv2.inRange(hsv_image, light_orange, dark_orange)

    # Set a white range
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)

    # Apply the white mask
    mask_white = cv2.inRange(hsv_image, light_white, dark_white)

    # Combine the two masks
    final_mask = mask + mask_white
    result = cv2.bitwise_and(image, image, mask=final_mask)

    # Clean up the segmentation using a blur
    blur = cv2.GaussianBlur(result, (7, 7), 0)
    return blur

results = [segment_image(i) for i in emma_images]

for i in range(5):
    plt.figure(figsize=(15,20))
    plt.subplot(1, 2, 1)
    plt.imshow(emma_images[i])
    plt.subplot(1, 2, 2)
    plt.imshow(results[i])
    plt.show()


