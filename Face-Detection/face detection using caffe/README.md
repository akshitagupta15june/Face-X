
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
ap.add_argument("-i", "--image", required=True,help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```
We have three required arguments:
- `--image` : The path to the input image.
- `--prototxt` : The path to the Caffe prototxt file.
- `--model` : The path to the pretrained Caffe model.

An optional argument, `--confidence` , can overwrite the default threshold of 0.5 

Load the model and create a blob from the image:

```
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
```
- Load the model using  `--prototxt`  and `--model`  file paths and store the model as net.
- Then load the image extract the dimensions  and create a blob.
- The `dnn.blobFromImage`  takes care of pre-processing which includes setting the blob  dimensions and normalization.

Next, apply face detection:

```
# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
```
To detect faces, pass the blob  through the net and from there  loop over the detections  and draw boxes around the detected faces:

```
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
```
We extract the confidence  and compare it to the confidence threshold . We perform this check to filter out weak detections.If the confidence meets the minimum threshold, we proceed to draw a rectangle and along with the probability of the detection.To accomplish this, we first calculate the (x, y)-coordinates of the bounding box ,then build our confidence text  string  which contains the probability of the detection.
In case the our text  would go off-image (such as when the face detection occurs at the very top of an image), we shift it down by 10 pixels.
From there we loop back for additional detections following the process again. If no detections  remain, we’re ready to show our output image  on the screen.

## Face detection in images with OpenCV result

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
