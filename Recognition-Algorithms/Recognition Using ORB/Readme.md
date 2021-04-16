## Feature recognition and matching using ORB algorithm
In this script, we would use the **ORB(Oriented FAST Rotated Brief)** algorithm of `Open CV` for recognition and matching the features of image.

ORB is a fusion of the FAST keypoint detector and BRIEF descriptor with some added features to improve the performance. FAST is Features from the Accelerated Segment Test used to detect features from the provided image. It also uses a pyramid to produce multiscale features. Now it doesn’t compute the orientation and descriptors for the features, so this is where BRIEF comes in the role.

## Dependencies
- Numpy
- OpenCV
## Usage
User has to provide the path of tha image. Than the image will be resized and passed through filters to make it appropiate for using the ORB algorithm. After this ORB algorithm is used to detect the feature of the face. After that by adding Scale Invariance and Rotational Invariance to the input image we create a test image. We use BruteForce matcher to match the feature of the 2 images.

## Sample Input/Output
<img src="https://i.ibb.co/6Y8Z04s/Robert-input.png" width=400/> <img src="https://i.ibb.co/J7z8nqT/Robert-feature.png" width=400/>
<img src="https://i.ibb.co/Kmfj2nL/Robert-Feaure-Matching.png"/>

## Author
[Shubham Gupta](https://github.com/ShubhamGupta577)
