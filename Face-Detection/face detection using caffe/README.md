
# Face Detection Using Caffe

## About

To perform fast, accurate face detection with OpenCV using a pre-trained deep learning face detector model shipped with the library.

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It has a huge applications 

## Files included

- The `source code`
- The `Caffe prototxt` files for deep learning face detection (defines model architecture)
- The `Caffe weight` files used for deep learning face detection (contains the weights of actual layers)
- The `example images` 

## Face detection in images

Download  `detect_faces.py` , `deploy.prototxt.txt` , `res10_300x300_ssd_iter_140000.caffemodel` and the input image .

**Command used:**

  >  *$ python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel*

In this given image , face is detected with 74% confidence using OpenCV deep learning face detection. 

![Example 1](outputs/deep_learning_face_detection_example_01.jpg)

**Command used:**

   >  *$ python detect_faces.py --image iron_chic.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel*
   
In this another example , the OpenCV DNN Face detector successfully finds all  the three faces.

![Example 2](outputs/deep_learning_face_detection_example_02.jpg)

## Face detection in video and webcam:

Download `detect_faces_video.py` , `deploy.prototxt.txt` , `res10_300x300_ssd_iter_140000.caffemodel` and run the deep learning OpenCV face detector with a webcam feed.

**Command used:**

   >  *$ python detect_faces_video.py --prototxt deploy.prototxt.txt  --model res10_300x300_ssd_iter_140000.caffemodel*

![Video](outputs/deep_learning_face_detection_opencv.gif)
