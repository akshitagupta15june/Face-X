# Pencil Sketch In Python Using OpenCV
## OpenCV

OpenCV is an open source computer vision and machine learning software library. It is a BSD-licence product thus free for both business and academic purposes.The Library provides more than 2500 algorithms that include machine learning tools for classification and clustering, image processing and vision algorithm, basic algorithms and drawing functions, GUI and I/O functions for images and videos. Some applications of these algorithms include face detection, object recognition, extracting 3D models, image processing, camera calibration, motion analysis etc.

OpenCV is written natively in C/C++. It has C++, C, Python and Java interfaces and supports Windows, Linux, Mac OS, iOS, and Android. OpenCV was designed for computational efficiency and targeted for real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing.

## Pencil Sketch in OpenCV

OpenCV 3 comes with a pencil sketch effect right out of the box. The cv2.pencilSketch function uses a domain filter introduced in the 2011 paper Domain transform for edge-aware image and video processing, by Eduardo Gastal and Manuel Oliveira. For customizations, other filters can also be developed.

###  Libraries Used

#### imread()

cv2.imread() method loads an image from the specified file. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format) then this method returns an empty matrix.
Note: The image should be in the working directory or a full path of image should be given.

All three types of flags are described below:

cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.

cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.

cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.

#### cvtColor() 

cv2.cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV.

Syntax: cv2.cvtColor(src, code[, dst[, dstCn]])

Parameters:

src: It is the image whose color space is to be changed.

code: It is the color space conversion code.

dst: It is the output image of the same size and depth as src image. It is an optional parameter.

dstCn: It is the number of channels in the destination image. If the parameter is 0 then the number of the channels is derived automatically from src and code. It is an optional parameter.

Return Value: It returns an image.

#### bitwise_not()

To make brighter regions lighter and lighter regions darker so that we could find edges to create a pencil sketch.

#### GaussianBlur()

In Gaussian Blur operation, the image is convolved with a Gaussian filter instead of the box filter. The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced. It also smoothens or blurs the image.

You can perform this operation on an image using the Gaussianblur() method of the imgproc class. Following is the syntax of this method −

GaussianBlur(src, dst, ksize, sigmaX)

This method accepts the following parameters −

src − A Mat object representing the source (input image) for this operation.

dst − A Mat object representing the destination (output image) for this operation.

ksize − A Size object representing the size of the kernel.

sigmaX − A variable of the type double representing the Gaussian kernel standard deviation in X direction.

#### dodgeV2()

It is used to divide the grey-scale value of image by the inverse of blurred image which highlights the sharpest edges.

### Results Obtained

![pencil_sketch1.png](attachment:pencil_sketch1.png)

![pencil_sketch2.png](attachment:pencil_sketch2.png)

![pencil_sketch3.png](attachment:pencil_sketch3.png)


```python

```
