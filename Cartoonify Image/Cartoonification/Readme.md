# Cartooning an Image using OpenCV – Python
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/preview-removebg.png" height="400px" align="left"/><img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/abe_toon.png"  height="400px" width="700px" align="top"/>

## Process of converting an image to a cartoon

- To convert an image to a cartoon, multiple transformations are done.
- Convert the image to a Grayscale image. Yes, similar to the old day’s pictures.!
- The Grayscale image is smoothened
- Extract the edges in the image
- Form a colour image and mask it with edges.
- This creates a beautiful cartoon image with edges and lightened colour of the original image.

### 1. Detecting and emphasizing edges
- Convert the original color image into grayscale
- Using adaptive thresholding to detect and emphasize the edges in an edge mask. 
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
  - <img src="https://github.com/Vi1234sh12/Face-X/blob/master/Cartoonify%20Image/Cartoonification/Threshold.jpg" height="300px" align="right"/>
 
 #### The next step is to apply `cv2.adaptiveThreshold()function`. As the parameters for this function we need to define:

   -  max value which will be set to 255
   - `cv2.ADAPTIVE_THRESH_MEAN_C `: a threshold value is the mean of the neighbourhood area.
   - `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` : a threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
   - `Block Size` – It determents the size of the neighbourhood area.
   - `C `– It is just a constant which is subtracted from the calculated mean (or the weighted mean).
### 2. Image filtering
- Apply a bilateral filter to create homogeneous colors on the image. 
### 3. Creating a cartoon effect
- Use a bitwise operation to combine the processed color image with the edge mask image.
### 4. Creating a cartoon effect using color quantization

## Steps to develop Image Cartoonifier
- Step 1: Importing the required modules
- Step 2: Building a File Box to choose a particular file
- Step 3: How is an image stored?
- Step 4: Transforming an image to grayscale
- Step 5: Smoothening a grayscale image
- Step 6: Retrieving the edges of an image
- Step 7: Giving a Cartoon Effect
- Step 8: Result 

### Issue #53 Cartoonifying a face image

In this notebook I tried to cartoonify uploaded image and video captured via webcam. You can use both here . If one's camera is enabled it will 
run via webcam.
If it is disabled it will want image path.


## Original Image
<img src="dicaprio.jpg" height="300px" >

## Cartoonified
<img src="Cartoonified_image.jpg" height="300px" >


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
