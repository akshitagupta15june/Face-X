# Face Morphing
This is a tool which creates a morphing effect. It takes two facial images as input and returns morphing from the first image to the second.

## Problem Statement:
Input: Tow images containing human faces(Image I1 and Image I2)

Output: A fluid transformation video transitioning from I1 to I2

Goal: The transition should be smooth and the intermediate frames should be as realistic as possible.

## Requirements
```
numpy
scikit_image
opencv_python
Pillow
skimage
dlib
```

# Example:
![face morph](https://github.com/sudipg4112001/Face-X/blob/master/Awesome-face-operations/Face-Morphing/Images/images.jpg)

## Steps:
- Provide two images in Images folder
- Generating a morphing animation video sequence
```
python3 code/__init__.py --img1 images/aligned_images/jennie.png --img2 images/aligned_images/rih.png --output output.mp4
```
- Run Face_Morpher.py above on your aligned face images with arg --img1 and --img2

## Features:
1. Detect and auto align faces in images (Optional for face morphing)
2. Generate corresponding features points between the two images using Dlib's Facial Landmark Detection
3. Calculate the triangular mesh with Delaunay Triangulation for each intermediate shape
4. Warp the two input images towards the intermediate shape, perform cross-dissolve and obtain intermediate images each frame
