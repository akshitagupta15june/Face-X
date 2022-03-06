# Face Morphing
This is a tool which creates a morphing effect. It takes two facial images as input and returns morphing from the first image to the second.
A user can input two images containing human faces(Image I1 and Image I2). 
The corresponding features points between the two images are generated using Dlib's Facial Landmark Detection. The triangular mesh for each intermediate shape is calculated with Delaunay Triangulation. The intermediate images for each frame are obtained by Warpping the two input images towards the intermediate shape and performing cross-dissolve. 
The output is a fluid transformation video transitioning from I1 to I2
The goal of this tool is that the transition should be smooth and the intermediate frames should be as realistic as possible.


<!-- ## Requirements
```
numpy
scikit_image
opencv_python
Pillow
skimage
dlib
``` -->

# Example:
![face morph](https://github.com/sudipg4112001/Face-X/blob/master/Awesome-face-operations/Face-Morphing/Images/images.jpg)

## Steps:
- Provide two images in Images folder
- Generating a morphing animation video sequence
```
python3 code/__init__.py --img1 images/aligned_images/jennie.png --img2 images/aligned_images/rih.png --output output.mp4
```
- Run Face_Morpher.py above on your aligned face images with arg --img1 and --img2


