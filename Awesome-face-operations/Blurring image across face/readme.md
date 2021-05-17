# Blurring Image Across Face <img src="https://img.icons8.com/color/48/000000/blur.png"/> 

<p align="center">
  <img style="text-align: center;" src="https://user-images.githubusercontent.com/64009389/111880150-51884080-89b2-11eb-80f1-12a1d8e53941.gif" alt="giphy" style="zoom:50%;" />
</p>

## Abstract <img src="https://img.icons8.com/color/30/000000/help--v1.png"/>

Blur Face makes it fast and simple to anonymize faces in your photos.
When processing an image, we are often interested in identifying objects represented within it so that we can perform some further analysis of these objects e.g. by counting them, measuring their sizes, etc. An important concept associated with the identification of objects in an image is that of edges: the lines that represent a transition from one group of similar pixels in the image to another different group. One example of an edge is the pixels that represent the boundaries of an object in an image, where the background of the image ends and the object begins.

When we blur an image, we make the color transition from one side of an edge in the image to another smooth rather than sudden. The effect is to average out rapid changes in pixel intensity. The blur, or smoothing, of an image removes “outlier” pixels that may be noise in the image. Blurring is an example of applying a low-pass filter to an image. In computer vision, the term “low-pass filter” applies to removing noise from an image while leaving the majority of the image intact. A blur is a very common operation we need to perform before other tasks such as edge detection. There are several different blurring functions in the skimage.filters module, so we will focus on just one here, the Gaussian blur.

Background blurring is most often seen as a feature of portrait mode in phone cameras. Another example is zoom and other online platforms that blur the background and not the face. In this model, we provide you with a small code to try this effect out, especially blurring the face.



## Requirements <img src="https://img.icons8.com/color/30/000000/settings.png"/>

- Python
- OpenCV



## Quick Start <img src="https://img.icons8.com/color/30/000000/google-code.png"/>

- Clone the Repository from [Here](https://github.com/akshitagupta15june/Face-X.git)
- Change the Directory: `cd "Blurring image across face"` or `cd Blurring\ image\ across\ face/`
- Run  `main.py`

  ##### Note:  This code might show error in VScode. PyCharm and jupyter notebook work fine.



## Result <img src="https://img.icons8.com/color/30/000000/image.png"/>

![](https://github.com/smriti1313/Face-X/blob/master/Blurring%20image%20across%20face/output.png)

