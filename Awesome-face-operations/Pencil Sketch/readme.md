# Pencil Sketch In Python Using OpenCV
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Pencil%20Sketch/Pencil_Image/book-pencil.png" weight="400px" height="400px"/><img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Pencil%20Sketch/Pencil_Image/girl_pencil.jpg" width="370px" height="400px" align="right"/>
## OpenCV

OpenCV is an open source computer vision and machine learning software library. It is a BSD-licence product thus free for both business and academic purposes.The Library provides more than 2500 algorithms that include machine learning tools for classification and clustering, image processing and vision algorithm, basic algorithms and drawing functions, GUI and I/O functions for images and videos. Some applications of these algorithms include face detection, object recognition, extracting 3D models, image processing, camera calibration, motion analysis etc.

OpenCV is written natively in C/C++. It has C++, C, Python and Java interfaces and supports Windows, Linux, Mac OS, iOS, and Android. OpenCV was designed for computational efficiency and targeted for real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing.

</p>

<p style="clear:both;">
<img alt="Layer5 Service Mesh Configuration Management" src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Pencil%20Sketch/Pencil_Image/pencil2.png"  style="margin-right;margin-bottom;" width="45%" align="right"/></a>
<h2> Pencil Sketch in OpenCV</h2>
OpenCV 3 comes with a pencil sketch effect right out of the box. The `cv2.pencilSketch` function uses a domain filter introduced in the 2011 paper Domain transform for edge-aware image and video processing, by Eduardo Gastal and Manuel Oliveira. For customizations, other filters can also be developed.
<br /><br /><br />
</p>

##   Libraries Used

 ### 1] imread()

`cv2.imread()` method loads an image from the specified file. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format) then this method returns an empty matrix.
Note: The image should be in the working directory or a full path of image should be given.

All three types of flags are described below:

- `cv2.IMREAD_COLOR:` It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.

- `cv2.IMREAD_GRAYSCALE:` It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.

- `cv2.IMREAD_UNCHANGED:` It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.

### 2] cvtColor()

`cv2.cvtColor()` method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV.

```
 cv2.cvtColor(src, code[, dst[, dstCn]])
```
Parameters:

- `src:` It is the image whose color space is to be changed.

- `code:` It is the color space conversion code.

- `dst:` It is the output image of the same size and depth as src image. It is an optional parameter.

- `dstCn:` It is the number of channels in the destination image. If the parameter is 0 then the number of the channels is derived automatically from src and code. It is an optional parameter.

Return Value: It returns an image.

 ### 3] bitwise_not()

To make brighter regions lighter and lighter regions darker so that we could find edges to create a pencil sketch.

 ### 4] GaussianBlur()

In Gaussian Blur operation, the image is convolved with a Gaussian filter instead of the box filter. The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced. It also smoothens or blurs the image.

You can perform this operation on an image using the `Gaussianblur()` method of the `imgproc` class. Following is the syntax of this method −

`GaussianBlur(src, dst, ksize, sigmaX)`

This method accepts the following parameters −

- `src` − A Mat object representing the source (input image) for this operation.

- `dst` − A Mat object representing the destination (output image) for this operation.

- `ksize` − A Size object representing the size of the kernel.

- `sigmaX` − A variable of the type double representing the Gaussian kernel standard deviation in X direction.

 ### 5] dodgeV2()

It is used to divide the grey-scale value of image by the inverse of blurred image which highlights the sharpest edges.

## Using OpenCV and Python, an RGB color image can be converted into a pencil sketch in four simple steps:
- Convert the RGB color image to grayscale.
- Invert the grayscale image to get a negative.
- Apply a Gaussian blur to the negative from step 2.
- Blend the grayscale image from step 1 with the blurred negative from step 3 using a color dodge.

### Step 1: Convert the color image to grayscale

This should be really easy to do even for an OpenCV novice. Images can be opened with `cv2.imread` and can be converted between color spaces with `cv2.cvtColor`. Alternatively, you can pass an additional argument to `cv2.imread` that specifies the color mode in which to open the image.

```
import cv2

img_rgb = cv2.imread("img_example.jpg")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
```
### Step 2: Obtain a negative

A negative of the image can be obtained by "inverting" the grayscale value of every pixel. Since by default grayscale values are represented as integers in the range [0,255] (i.e., precision CV_8U), the "inverse" of a grayscale value x is simply 255-x:

```
img_gray_inv = 255 - img_gray
```
### Step 3: Apply a Gaussian blur

A Gaussian blur is an effective way to both reduce noise and reduce the amount of detail in an image (also called smoothing an image). Mathematically it is equivalent to convolving an image with a Gaussian kernel. The size of the Gaussian kernel can be passed to cv2.GaussianBlur as an optional argument ksize. If both sigmaX and sigmaY are set to zero, the width of the Gaussian kernel will be derived from ksize:

```
img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),sigmaX=0, sigmaY=0)
```
### Step 4: Blend the grayscale image with the blurred negative
This is where things can get a little tricky. Dodging and burning refer to techniques employed during the printing process in traditional photography. In the good old days of traditional photography, people would try to lighten or darken a certain area of a darkroom print by manipulating its exposure time. Dodging lightened an image, whereas burning darkened it.

Modern image editing tools such as Photoshop offer ways to mimic these traditional techniques. For example, color dodging of an image A with a mask B is implemented as follows:

```
((B[idx] == 255) ? B[idx] : min(255, ((A[idx] << 8) / (255-B[idx]))))
```
This is essentially dividing the grayscale (or channel) value of an image pixel A[idx] by the inverse of the mask pixel value B[idx], while making sure that the resulting pixel value will be in the range [0,255] and that we do not divide by zero. We could translate this into a naïve Python function that accepts two OpenCV matrices (an image and a mask) and returns the blended mage:

```
import cv2
img=cv2.imread("img.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_invert = cv2.bitwise_not(img_gray)
img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)
final_img = dodgeV2(img_gray, img_smoothing)
cv2.imshow('result',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Instead, we should realize that the operation <<8 is the same as multiplying the pixel value with the number 2^8=256, and that pixel-wise division can be achieved with `cv2.divide`. An improved version of the dodging function could thus look like this:

```
def dodgeV2(image, mask):
  return cv2.divide(image, 255-mask, scale=256)
```
The function `dodgeV2` produces the same result as dodgeNaive but is orders of magnitude faster. In addition, `cv2.divide` automatically takes care of the division by zero, making the result 0 where 255-mask is zero. A burning function can be implemented analogously:

```
def burnV2(image, mask):
  return 255 – cv2.divide(255-image, 255-mask, scale=256)
```
#### now complete the pencil sketch transformation:
```
img_blend = dodgeV2(img_gray, img_blur)
cv2.imshow("pencil sketch", img_blend)
```
#### Results Obtained

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Pencil%20Sketch/Pencil_Image/pencil4.png"/>





```python

```
