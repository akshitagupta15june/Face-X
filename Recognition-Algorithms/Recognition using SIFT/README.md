# Facial-recognition-using-SIFT
This is an experimental facial recognition project by matching the features extracted using SIFT. 

### Dependencies
1. numpy
2. opencv-contrib-python version 3.4.2.16
3. pnslib

### Brief
Two images are taken as input. For now, only images consisting of a single face are considered. The images are passed through a face detection algorithm. For face detection, we use OpenCV's haarcascade classifier. After the faces are detected, we crop out the region of interests from the images and pass it on to the feature extraction algorithm.

For feature extraction ,we use the SIFT algorithm in OpenCV.SIFT produces a list of good features for each image. Each of this features is a 128 dimensional vector. We use a BruteForce matcher to match the features of the 2 images. For each feature in each image, we consider the 2 most similar features in the other image and filter out the good matches among them. Good matches are those matches which are atmost 0.75 times closer than the second most similar feature.

After feature matching using the BruteForce matcher, the decision of Match or No-Match is done based on the number of good matches for the image pair. This is a crude way of deciding, still worth being a starting point. 

### Screenshot

![capture](facial_recognition.png)

### References
1. Face detection using Haar Cascades : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
2. Introduction to SIFT in OpenCV : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
3. Feature matching in OpenCV : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
4. BruteForce OpenCV tutorial (future ref) : https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/
5. Feature matching + homography (future ref) : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
