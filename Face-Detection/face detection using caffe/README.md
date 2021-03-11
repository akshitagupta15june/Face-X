
# Face Detection Using Caffe

## About

To perform fast, accurate face detection with OpenCV using a pre-trained deep learning face detector model shipped with the library.

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It has a huge applications 

## Files included

- The `source code`
- The `Caffe prototxt` files for deep learning face detection (defines model architecture)
- The `Caffe weight` files used for deep learning face detection (contains the weights of actual layers)
- The `example images` 

## Face Detection in Images using OpenCV and Deep Learning

**Code for Face Detection**

These are following lines of code from the file `detect_face.py` :

```
# import the necessary packages
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```
We have three required arguments:

. `--image` : The path to the input image.
. `--prototxt` : The path to the Caffe prototxt file.
. `--model` : The path to the pretrained Caffe model.

## Face detection in images

Download  `detect_faces.py` , `deploy.prototxt.txt` , `res10_300x300_ssd_iter_140000.caffemodel` and the input image .

### Image 1

**Command used:**

  >  *$ python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel*

In this given image , face is detected with 74% confidence using OpenCV deep learning face detection. 

![Example 1](outputs/deep_learning_face_detection_example_01.jpg)

### Image 2

**Command used:**

   >  *$ python detect_faces.py --image iron_chic.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel*
   
In this another example , the OpenCV DNN Face detector successfully finds all  the three faces.

![Example 2](outputs/deep_learning_face_detection_example_02.jpg)

## Face detection in video and webcam:

Download `detect_faces_video.py` , `deploy.prototxt.txt` , `res10_300x300_ssd_iter_140000.caffemodel` and run the deep learning OpenCV face detector with a webcam feed.

**Command used:**

   >  *$ python detect_faces_video.py --prototxt deploy.prototxt.txt  --model res10_300x300_ssd_iter_140000.caffemodel*

![Video](outputs/deep_learning_face_detection_opencv.gif)
