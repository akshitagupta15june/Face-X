# Awesome-face-operations

![image](https://user-images.githubusercontent.com/78999467/112627758-1bd3d380-8e5a-11eb-9c41-39a98e11c1c1.png)

# Face Morphing
This is a tool that creates a morphing effect. It takes two facial images as input and returns morphing from the first image to the second.
### Example:
![face morph](https://github.com/sudipg4112001/Face-X/blob/master/Awesome-face-operations/Face-Morphing/Images/images.jpg)
### Steps:
```diff
- Find point-to-point correspondences between the two images.
- Find the Delaunay Triangulation for the average of these points.
- Using these corresponding triangles in both initial and final images, perform Warping and Alpha Blending and obtain intermediate images. 
```

# Converting an image into a ghost image.

Used OpenCV and Numpy to convert an image into a ghost image.

### Steps:
```diff
- Imported the required libraries ( Numpy, Matplotlib, Cv2)
- Read the input image using cv2
```
### Methods applied Using Cv2
```diff
- Used Bilateral Filter
- Used Median Blur
- Used Adaptive Threshold
- Used Bitwise Xor
- Finally converted the image into a ghost image
```

### Original Image
<img src="Images/photo.jpg" height="300px">

### Ghost Image
<img src="Images/Ghost Photo.jpg" height="300px">

# Pencil Sketch In Python Using OpenCV
### OpenCV

OpenCV is an open-source computer vision and machine learning software library. It is a BSD-licence product thus free for both business and academic purposes. OpenCV is written natively in C/C++. It has C++, C, Python, and Java interfaces and supports Windows, Linux, Mac OS, iOS, and Android. OpenCV was designed for computational efficiency and targeted for real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing.

### Pencil Sketch in OpenCV

OpenCV 3 comes with a pencil sketch effect right out of the box. The cv2.pencilSketch function uses a domain filter introduced in the 2011 paper Domain transform for edge-aware image and video processing, by Eduardo Gastal and Manuel Oliveira. For customizations, other filters can also be developed.

###  Libraries Used

#### imread()
cv2.imread() method loads an image from the specified file. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format) then this method returns an empty matrix.
#### cvtColor()
cv2.cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV. 
#### bitwise_not()
To make brighter regions lighter and lighter regions darker so that we could find edges to create a pencil sketch.
#### GaussianBlur()
In the Gaussian Blur operation, the image is convolved with a Gaussian filter instead of the box filter. The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced. It also smoothens or blurs the image. You can perform this operation on an image using the Gaussianblur() method of the imgproc class.
#### dodgeV2()
It is used to divide the grey-scale value of the image by the inverse of the blurred image which highlights the sharpest edges.
### Results Obtained

![pencil_sketch1.png](attachment:pencil_sketch1.png)

![pencil_sketch2.png](attachment:pencil_sketch2.png)

![pencil_sketch3.png](attachment:pencil_sketch3.png)

<h1> Image Segmentation Using Color space and Opencv</h1>
<h2>Introduction</h2>
<p>
The process of partitioning a digital image into multiple segments is defined as image segmentation. Segmentation aims to divide an image into regions that can be more representative and easier to analyze.</p>

<h2>What are color spaces?</h2>
<p>Basically, Color spaces represent color through discrete structures (a fixed number of whole number integer values), which is acceptable since the human eye and perception are also limited. Color spaces are fully able to represent all the colors that humans are able to distinguish between.</p>

 
## Steps followed for implementation
```diff
- Converted the image into HSV
- Choosing swatches of the desired color, In this, shades of light and dark orange have been taken.
- Applying an orange shade mask to the image
- Adding the second swatches of color, Here shades of white were chosen i.e light and dark shades
- Apply the white mask onto the image
- Now combine the two masks, Adding the two masks together results in 1 value wherever there is an orange shade or white shade.
- Clean up the segmentation using a blur 
```

 
 <p>
 <h2>Default  image in BGR color space</h2>
 <img src="images\BGR_IMAGE.PNG">
 
 <h2>Image converted to RGB color space</h2>
 <img src="images\RBG_IMAGE.PNG">
 
 <h2>Image converted to GRAY color space</h2>
 <img src="images\GRAY_IMAGE.PNG">
 
 <h2>Image converted to HSV color space</h2>
 <img src="images\HSV_IMAGE.PNG">
 </p>
 
 <p>
 <h2>Segmented images</h2>
 <img src="images\demo1.PNG">
 <img src="images\demo2.PNG">
 </p>


```python

```

