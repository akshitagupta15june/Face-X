# Facial Rcognition using SURF 

Intoduction about SURF:-
SURF stands for Speeded-Up Robust Features.Basically SURF is an algorithm used in computer vision. It is a patented local feature detector and descriptor. We use SURF for various tasks like object recognition, image registration, classification or  3D reconstruction .

And here we are experimenting with SURF in context of facial recognition.

### Library Requirements
1. opencv-contrib-python
2. numpy
3. matplotlib

### So what been done?
1. First required libraries have been imported.
2. Then two images have been imported to work on.
3. Converted both the images in grayscale.
4. Take out the features i.e. keypoints and descriptors of both the images
5. Then we proceed to feature matching.
6. For feature matching two algorithms are used:-      i. BruteForceMatch
     ii. FLANN(Fast Approximate Nearest Neighbour Search          Algorithm)

### References
1. SURF implementation: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html

2. Feature Matching(BruteForce and FLANN): https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html