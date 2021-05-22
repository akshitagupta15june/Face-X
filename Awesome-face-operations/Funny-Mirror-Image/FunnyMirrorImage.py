import cv2
import numpy as np
import math
from vcam import vcam,meshGen
import matplotlib.pyplot as plt


# Reading the input image. Pass the path of image you would like to use as input image.
img = cv2.imread("dora.png")
H,W = img.shape[:2]


# Creating the virtual camera object
c1 = vcam(H=H,W=W)

# Creating the surface object
plane = meshGen(H,W)

# We generate a mirror where for each 3D point, its Z coordinate is defined as Z = 20*exp^((x/w)^2 / 2*0.1*sqrt(2*pi))

plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
pts3d = plane.getPlane()

pts2d = c1.project(pts3d)
map_x,map_y = c1.getMaps(pts2d)

output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)

'''plt.subplot(1, 2,1)
plt.title("Funny Mirror")
cv2.imshow(cv2.cvtColor(np.hstack((img,output)), cv2.COLOR_BGR2RGB))'''

cv2.imshow("Original Image",img)
cv2.imshow("Funy Mirror Image",output)
cv2.waitkey(0)
