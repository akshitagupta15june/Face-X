# Cartoonify an Image with OpenCV in Python
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/Assets/preview-removebg.png" height="400px" align="left"/><img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/Assets/abe_toon.png"  height="400px" width="600px" align="top"/>

## Process of converting an image to a cartoon

- To convert an image to a cartoon, multiple transformations are done.
- Convert the image to a Grayscale image. Yes, similar to the old day’s pictures.!
- The Grayscale image is smoothened
- Extract the edges in the image
- Form a colour image and mask it with edges.
- This creates a beautiful cartoon image with edges and lightened colour of the original image.

## How to start

- Fork and Clone the repository using-
```
git clone https://github.com/akshitagupta15june/Face-X.git
```
- Create virtual environment-
```
- python -m venv env
- source env/bin/activate (Linux)
- env\Scripts\activate (Windows)
```
- Install dependencies
- Go to project directory
```
- cd Cartoonify Image
```
- Open Terminal
```
python cartoonify_without_GUI.py --image IMAGE_PATH
```
### 1. Detecting and emphasizing edges
- Convert the original color image into grayscale
- Using adaptive`thresholding` to detect and `emphasize` the edges in an edge mask. 
- Apply a median blur to reduce image noise.
  - `-->` To produce accurate carton effects, as the first step, we need to understand the difference between a common digital image and a cartoon image.At the first glance we can clearly see two major differences.
   - The first difference is that the colors in the cartoon image are more homogeneous as compared to the normal image.
   - The second difference is noticeable within the edges that are much sharper and more pronounced in the cartoon.
   -  Let’s begin by importing the necessary libraries and loading the input image.
```
      import cv2
      import numpy as np
```
   - Now, we are going to load the image.
```
img = cv2.imread("Superman.jpeg")
cv2_imshow(img)
```
  - The next step is to detect the edges. For that task, we need to choose the most suitable method. Remember, our goal is to detect clear edges. There are several edge detectors that we can pick. Our first choice will be one of the most common detectors, and that is the `Canny edge detector`. But unfortunately, if we apply this detector we will not be able to achieve desirable results. We can proceed with Canny, and yet you can see that there are too many details captured. This can be changed if we play around with Canny’s input parameters (numbers 100 and 200).
  - Although Canny is an excellent edge detector that we can use in many cases in our code we will use a threshold method that gives us more satisfying results. It uses a threshold pixel value to convert a grayscale image into a binary image. For instance, if a pixel value in the original image is above the threshold, it will be assigned to 255. Otherwise, it will be assigned to 0 as we can see in the following image.
  - <img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/Assets/Threshold.jpg" height="300px" align="right"/>
 
 #### The next step is to apply `cv2.adaptiveThreshold()function`. As the parameters for this function we need to define:

   -  max value which will be set to 255
   - `cv2.ADAPTIVE_THRESH_MEAN_C `: a threshold value is the mean of the neighbourhood area.
   - `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` : a threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
   - `Block Size` – It determents the size of the neighbourhood area.
   - `C `– It is just a constant which is subtracted from the calculated mean (or the weighted mean).
```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
```
```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_1 = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
```
   - <img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/Assets/filters_tutorial_02.png" align ="right" height="400px"/>
### 2. Image filtering
- Apply a bilateral filter to create homogeneous colors on the image. 
### 3. Creating a cartoon effect
- Use a bitwise operation to combine the processed color image with the edge mask image.
- Our final step is to combine the previous two: We will use  `cv2.bitwise_and()` the function to mix edges and the color image into a single one
```
 cartoon = cv2.bitwise_and(color, color, mask=edges)
 cv2_imshow(cartoon)
```
### 4. Creating a cartoon effect using color quantization
 - Another interesting way to create a cartoon effect is by using the color quantization method. This method will reduce the number of colors in the image and that will create a cartoon-like effect. We will perform color quantization by using the K-means clustering algorithm for displaying output with a limited number of colors. First, we need to define `color_quantization()` function.
 ```
 def color_quantization(img, k):
# Defining input data for clustering
  data = np.float32(img).reshape((-1, 3))
# Defining criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying cv2.kmeans function
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result
 ```
 - Different values for K will determine the number of colors in the output picture. So, for our goal, we will reduce the number of colors to 7. Let’s look at our results.
 ```
img_1 = color_quantization(img, 7)
cv2_imshow(img_1)
```

## Steps to develop Image Cartoonifier

- Step 1: Importing the required modules
```
import cv2
import argparse
```
- Step 2: Transforming an image to grayscale
```
#converting an image to grayscale
grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
ReSized2 = cv2.resize(grayScaleImage, (960, 540))
#plt.imshow(ReSized2, cmap='gray')
```
   - Transforming an image to grayscale
      - `cvtColor(image, flag)` is a method in cv2 which is used to transform an image into the colour-space mentioned as ‘flag’. Here, our first step is to convert the image           into grayscale. Thus, we use the `BGR2GRAY` flag. This returns the image in grayscale. A grayscale image is stored as `grayScaleImage`.
       - After each transformation, we resize the resultant image using the resize() method in cv2 and display it using imshow() method. This is done to get more clear insights         into every single transformation step.
- Step 3: Smoothening a grayscale image
```
#applying median blur to smoothen an image
smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
ReSized3 = cv2.resize(smoothGrayScale, (960, 540))
#plt.imshow(ReSized3, cmap='gray')
```
   - Smoothening a grayscale image
     - To smoothen an image, we simply apply a blur effect. This is done using medianBlur() function. Here, the center pixel is assigned a mean value of all the pixels which fall under the kernel. In turn, creating a blur effect.
- Step 4: Retrieving the edges of an image
```
#retrieving the edges for cartoon effect
#by using thresholding technique
getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
  cv2.ADAPTIVE_THRESH_MEAN_C, 
  cv2.THRESH_BINARY, 9, 9)
ReSized4 = cv2.resize(getEdge, (960, 540))
#plt.imshow(ReSized4, cmap='gray')
```
   - Cartoon effect has two specialties:
     - Highlighted Edges
     - Smooth color
     - In this step, we will work on the first specialty. Here, we will try to retrieve the edges and highlight them. This is attained by the adaptive thresholding technique. The threshold value is the mean of the neighborhood pixel values area minus the constant C. C is a constant that is subtracted from the mean or weighted sum of the neighborhood pixels. Thresh_binary is the type of threshold applied, and the remaining parameters determine the block size.
- Step 5: Giving a Cartoon Effect
``` 
#masking edged image with our "BEAUTIFY" image
cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
ReSized6 = cv2.resize(cartoonImage, (960, 540))
#plt.imshow(ReSized6, cmap='gray')

```
### Results Obtained
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/Assets/result%20(2).jpg" hight="300px" width="700px"/>

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/Assets/boy.png" height="400px" align="left"/>
<p style="clear:both;">
<h1><a name="contributing"></a><a name="community"></a> <a href="https://github.com/akshitagupta15june/Face-X">Community</a> and <a href="https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md">Contributing</a></h1>
<p>Please do! Contributions, updates, <a href="https://github.com/akshitagupta15june/Face-X/issues"></a> and <a href=" ">pull requests</a> are welcome. This project is community-built and welcomes collaboration. Contributors are expected to adhere to the <a href="https://gssoc.girlscript.tech/">GOSSC Code of Conduct</a>.
</p>
<p>
Jump into our <a href="https://discord.com/invite/Jmc97prqjb">Discord</a>! Our projects are community-built and welcome collaboration. 👍Be sure to see the <a href="https://github.com/akshitagupta15june/Face-X/blob/master/Readme.md">Face-X Community Welcome Guide</a> for a tour of resources available to you.
</p>
<p>
<i>Not sure where to start?</i> Grab an open issue with the <a href="https://github.com/akshitagupta15june/Face-X/issues">help-wanted label</a>
</p>

**Open Source First**

 best practices for managing all aspects of distributed services. Our shared commitment to the open-source spirit push the Face-X community and its projects forward.</p>
