## Face-Swapping

This application lets you swap a face in one image with a face from another image. 

## Steps used for this project:

1. Taking two images – one as the source and another as a destination.
2. Using the dlib landmark detector on both these images. 
3. Joining the dots in the landmark detector to form triangles. 
4. Extracting these triangles
5. Placing the source image on the destination and smoothening the face

## Selecting Images

You can select any two images of your choice. Both the images are front-facing and are well lit.

## Using the dlib landmark detector on the images

Dlib is a python library that provides us with landmark detectors to detect important facial landmarks. These 68 points are important to identify the different features in both faces.

## Joining the dots in the landmark detector to form triangles for the source image.

To cut a portion of the face and fit it to the other we need to analyse the size and perspective of both the images. To do this, we will split the entire face into smaller triangles by joining the landmarks so that the originality of the image is not lost and it becomes easier to swap the triangles with the destination image.

## Extracting these triangles 

Once we have the triangles in source and destination the next step is to extract them from the source image.

## Placing the source image on the destination

Now, we can reconstruct the destination image and start placing the source image on the destination one.

### Example 1

![face swap sample](https://raw.githubusercontent.com/sudipg4112001/Face-X/master/Face-Swapping/Sample%20images/Face_swap_2.jpg)

### Example 2

![face swap sample](https://raw.githubusercontent.com/sudipg4112001/Face-X/master/Face-Swapping/Sample%20images/face_swap_1.png)
