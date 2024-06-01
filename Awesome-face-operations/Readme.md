# Awesome-face-operations

<img src="https://user-images.githubusercontent.com/78999467/112627758-1bd3d380-8e5a-11eb-9c41-39a98e11c1c1.png" alt="drawing" width="500"/>

# Face Morphing
This is a tool which creates a morphing effect. It takes two facial images as input and returns morphing from the first image to the second.
A user can input two images containing human faces(Image I1 and Image I2). The corresponding features points between the two images are generated using Dlib's Facial Landmark Detection. The triangular mesh for each intermediate shape is calculated with Delaunay Triangulation. The intermediate images for each frame are obtained by Warpping the two input images towards the intermediate shape and performing cross-dissolve. The output is a fluid transformation video transitioning from I1 to I2. The goal of this tool is that the transition should be smooth and the intermediate frames should be as realistic as possible.
Run Face_Morpher.py on your aligned face images with arg --img1 and --img2
```
python3 code/__init__.py --img1 images/aligned_images/jennie.png --img2 images/aligned_images/rih.png --output output.mp4
```

### Example:
<img src="https://github.com/sudipg4112001/Face-X/blob/master/Awesome-face-operations/Face-Morphing/Images/images.jpg" alt="drawing" width="500"/>


# Converting an image into a ghost image.
Uses OpenCV and Numpy to convert an image into a ghost image. This tool Imports the required libraries ( Numpy, Matplotlib, Cv2) and Reads the input image using cv2. Bilateral Filter, Median Blur, Adaptive Threshold, and Bitwise Xor methods are applied on the the image. The image is finally converted into a ghost image.

### Original Image
<img src="https://user-images.githubusercontent.com/78999467/112639805-c6eb8980-8e68-11eb-9312-312a5df65aa1.png" alt="drawing" width="300" height="300"/>

### Ghost Image
<img src="https://user-images.githubusercontent.com/78999467/112639825-cce16a80-8e68-11eb-9920-7d515ff158e4.png" alt="drawing" width="300" height="300"/>


# Pencil Sketch In Python Using OpenCV
OpenCV 3 comes with a pencil sketch effect right out of the box. The cv2.pencilSketch function uses a domain filter introduced in the 2011 paper Domain transform for edge-aware image and video processing, by Eduardo Gastal and Manuel Oliveira. For customizations, other filters can also be developed. First, the cv2.imread() method loads an image from the specified file. The cv2.cvtColor() method is used to convert the image from one color space to another. Then bitwise_not() is used to make brighter regions lighter and lighter regions darker so that we could find edges to create a pencil sketch. Next, The GaussianBlur() operation is used with the use of which, the image is convolved with a Gaussian filter instead of the box filter. The Gaussian filter is a low-pass filter that removes the high-frequency components and also smoothens or blurs the image. Finally, dodgeV2() is used to divide the grey-scale value of the image by the inverse of the blurred image which highlights the sharpest edges.

### Results Obtained
<img src="https://user-images.githubusercontent.com/78999467/112639271-2dbc7300-8e68-11eb-8c99-314d1bffa1b1.png" alt="drawing" width="300" height="300"/>
<img src="https://user-images.githubusercontent.com/78999467/112639296-344aea80-8e68-11eb-85a9-401529d63164.png" alt="drawing" width="300" height="300"/>
<img src="https://user-images.githubusercontent.com/78999467/112639322-3a40cb80-8e68-11eb-8a6e-266b923b038e.png" alt="drawing" width="300" height="300"/>


# Image Segmentation Using Color space and OpenCV
The process of partitioning a digital image into multiple segments is defined as image segmentation. Segmentation aims to divide an image into regions that can be more representative and easier to analyze.
The image is first converted into HSV. Then, swatches of the desired color are chosen and a mask of the chosen color is applied to the image. Next, swatches of the second color are chosen and a mask of the chosen color is applied to the image. Next the two masks are combined and the segmentation is cleaned up using a blur.

 ### Default  image in BGR color space
 <img src="https://user-images.githubusercontent.com/78999467/112638972-e59d5080-8e67-11eb-91a6-aff48f35c1c0.png" alt="drawing" width="500"/>

 ### Image converted to RGB color space
 <img src="https://user-images.githubusercontent.com/78999467/112638902-d3bbad80-8e67-11eb-9885-e7e2e367bb8c.png" alt="drawing" width="500"/>

 ### Image converted to GRAY color space
 <img src="https://user-images.githubusercontent.com/78999467/112638849-c4d4fb00-8e67-11eb-9d10-413da262d1d2.png" alt="drawing" width="500"/>
 
 ### Image converted to HSV color space
 <img src="https://user-images.githubusercontent.com/78999467/112638768-b38bee80-8e67-11eb-9f94-037ed3acf9ea.png" alt="drawing" width="500"/>


### Segmented images
<img src="https://user-images.githubusercontent.com/78999467/112638705-a2db7880-8e67-11eb-89f3-87f16f1ed8d2.png" alt="drawing" width="500"/>


# Blurring Images Across Face
An image seems more detailed if we can observe all the objects and their shapes accurately in it. For instance, an image with a face looks clear when we can identify eyes, ears, etc very clear. This shape of an object is due to its edges. So in blurring, we simply reduce the edge content and makes the transition from one color to the other very smooth.
Background blurring is most often seen as a feature of portrait mode in phone cameras. Another example is zoom and other online platforms that blur the background and not the face. In this model, we provide you with a small code to try this effect out, especially blurring the face.

### Result 
<img src="https://github.com/smriti1313/Face-X/blob/master/Blurring%20image%20across%20face/output.png" alt="drawing" width="500" height="300"/>


# Face Deblurring
Face deblurring operation is the task of estimating a clear image from its degraded blur image and recovering the sharp contents and textures. The aim of face deblurring is to restore clear images with more explicit structure and facial details. The face deblurring problem has attracted considerable attention due to its wide range of applications.

### Original Image
<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Awesome-face-operations/Face%20Deblurring/Images/Blurry%20Images/000001.png" alt="drawing" width="300" height="300"/>

### Colorful Sketch Filtered Image
<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Awesome-face-operations/Face%20Deblurring/Images/Clean%20Images/000001.png" alt="drawing" width="300" height="300"/>


# Generating Faces using GANs
In this operation, we define and train a DCGAN on a dataset of faces. The goal is to get a generator network to generate new images of faces that look as realistic as possible.
The operations are broken down into a series of tasks from loading in data to defining and training adversarial networks. This tool enables the user to visualize the results of the trained Generator to see how it performs; the generated samples should look like fairly realistic faces with small amounts of noise.
DataSet: [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train the adversarial networks.

### Some samples of generated faces:
<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Awesome-face-operations/Face%20Generation/GAN_Face_Generation/assets/generated_faces.png"/>

### To Generate Faces:
1. Just run the script ```dlnd_face_generation.ipynb```


# Gender Classification
it is a simple gender recognition system from facial image where we first detect faces from a scene using Haar Feature Based `Cascade Classifier` then introducing it to the model architecture. The face detection goal is achieved by OpenCV. We use Haarcascade and OpenCV to detect faces in a live webcam input stream. Then, we will retrain an inception v3 Artificial Neural Network to classify Male and Female faces. As training data, we are going to scrape some images from Bing Images search. Afterwards, we will use this slow inception v3 model to classify a big dataset of about 15'000 face images automatically, which we will then use to train a much faster Neural Network which will enhance the performance of the live classifier significantly.

### Outputs
<img src="https://user-images.githubusercontent.com/63206325/113521952-4bdb4f00-959d-11eb-9af6-36e422919f74.png" alt="drawing" width="300" height="300"/>

<img src="https://user-images.githubusercontent.com/63206325/113521963-5990d480-959d-11eb-8649-457005a0031e.png" alt="drawing" width="300" height="300"/>

<img src="https://user-images.githubusercontent.com/63206325/113521969-631a3c80-959d-11eb-80db-53381a3a35af.png" alt="drawing" width="300" height="300"/>

<img src="https://user-images.githubusercontent.com/63206325/113521972-6c0b0e00-959d-11eb-8a8d-bccc183e879b.png" alt="drawing" width="300" height="300"/>


# Glitter Filter
Converting an image into a glitter cartoon filtered image using OpenCv, Os, Matplotlib and Numpy. Uses Bilateral Filter followed by Median Blur followed by Adaptive Threshold, followed by Bitwise "or" between original image and image_rgb and at last used Bitwise "and" between image_edge and output of the above "bitwise or image" finally converted the image into "Glitter Cartoon Filtered" image

### Example:
<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Cartoonify-Image/Glitter%20Cartoon%20Filter/Images/Final_Output.jpg" alt="drawing" width="500" height="300"/>


## Image Stitching
The input to the stitching process is several images with overlapping areas. The output is a unification of these images. It is important to note that a full scene from the input image must be preserved in the process. To construct image stiching, we utilize computer vision and image processing techniques such as: keypoint detection and local invariant descriptors; keypoint matching; RANSAC; and perspective warping.

### Example:
<img src="https://github.com/sudipg4112001/Face-X/blob/master/Awesome-face-operations/Image-Stiching/Sample-img.jpg" alt="drawing" width="500"/>


# Multiple-Template-Matching
Template Matching is a method for searching and finding the location of a template image in a larger image. It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image. Several comparison methods are implemented in OpenCV.
- If input image is of size (WxH) and template image is of size (wxh), output image will have a size of (W-w+1, H-h+1). 
- Take it as the top-left corner of rectangle and take (w,h) as width and height of the rectangle. That rectangle is your region of template.
Suppose you are searching for an object which has multiple occurances, `cv2.minMaxLoc()` won’t give you all the locations. In that case, we will use thresholding. 

### Example:
<img src="https://user-images.githubusercontent.com/60208804/113759937-47e13580-9733-11eb-9c1c-c2acf373c8e6.jpg" alt="drawing" width="500"/>


# Real Time Age Prediction
Age detection is the process of automatically discerning the age of a person solely from a photo of their face. Typically, the first step is to detect faces in the input image/video stream. Then, we Extract the face Region of Interest (ROI), and apply the age detector algorithm to predict the age of the person.

### Outputs
Real Age = 35
<img src="https://user-images.githubusercontent.com/55057549/112677174-0ff40b80-8e72-11eb-96a6-e846adfb80be.PNG" alt="drawing" width="500" height="300"/>

Real Age = 85
<img src="https://user-images.githubusercontent.com/55057549/112677632-aaece580-8e72-11eb-9e4b-5f18d2a29aeb.PNG" alt="drawing" width="500" height="300"/>


# Style Transfer
Style transfer relies on separating the content and style of an image. Given one content image and one style image, the aim is to create a new, target image which should contain the desired content and style components:
* objects and their arrangement are similar to that of the **content image**
* style, colors, and textures are similar to that of the **style image**

### Example
<img src="https://github.com/KKhushhalR2405/Style-Transfer/blob/master/exp1/blonde.jpg" alt="drawing" width="300" height="300"/>
<img src="https://github.com/KKhushhalR2405/Style-Transfer/blob/master/exp1/final_image.png" alt="drawing" width="300" height="300"/>


## Template Detection
Template Matching is a method for searching and finding the location of a template image in a larger image. OpenCV comes with a function `cv2.matchTemplate()` for this purpose. It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image. Several comparison methods are implemented in OpenCV. (You can check docs for more details). It returns a grayscale image, where each pixel denotes how much does the neighbourhood of that pixel match with template.

### Example:
<img src="https://user-images.githubusercontent.com/60208804/113617568-1b1a1900-9674-11eb-8957-07e1977c7864.jpg" alt="drawing" width="500"/>


# Video-BG-Substraction
Background subtraction is a major preprocessing steps in many vision based applications. For example, consider the cases like visitor counter where a static camera takes the number of visitors entering or leaving the room, or a traffic camera extracting information about the vehicles etc. In all these cases, first you need to extract the person or vehicles alone. Technically, you need to extract the moving foreground from static background.
One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc. Here, you have an option of selecting whether shadow to be detected or not. If `detectShadows = True` (which is so by default), it detects and marks shadows, but decreases the speed. Shadows will be marked in gray color.

### Input:
<img src="https://user-images.githubusercontent.com/60208804/113537714-106d6e80-95f7-11eb-8590-7d7b12e7760b.jpg">

### Output:
<img src="https://user-images.githubusercontent.com/60208804/113537728-195e4000-95f7-11eb-8f3d-edcaf79ddc36.jpg" alt="drawing" width="500"/>


# More Awesome Face Operations That Can Be Added Here 
![Face_Alignment](https://raw.githubusercontent.com/YadiraF/PRNet/master/Docs/images/alignment.jpg "Facial Alignment Analysis")

### Face Detection
### Face Alignment
### Face Recognition
### Face Identification
### Face Verification
### Face Representation
### Face Alignment
### Face(Facial) Attribute & Face(Facial) Analysis
### Face Reconstruction
### Face 3D
### Face Tracking
### Face Clustering
### Face Super-Resolution
### Face Deblurring
### Face Hallucination
### Face Generation
### Face Synthesis
### Face Completion
### Face Restoration
### Face De-Occlusion
### Face Transfer
### Face Editing
### Face Anti-Spoofing
### Face Retrieval
### Face Application

---
## Pipelines

- [seetaface/SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine)
---
