When using OpenCV’s deep neural network module with Caffe models, you’ll need two sets of files:

- The **.prototxt** file(s) which define the model architecture (i.e., the layers themselves)
- The **.caffemodel** file which contains the weights for the actual layers

Open up a `detect_faces.py`  and see the following lines :

```
# import the necessary packages
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

Here we are importing our required packages and parsing command line arguments.

We have three required arguments:

- `--image` : The path to the input image.
- `--prototxt` : The path to the Caffe prototxt file.
- `--model` : The path to the pretrained Caffe model.

An optional argument, `--confidence` , can overwrite the default threshold of 0.5 if you wish.

From there lets load our model and create a blob from our image:

```
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
```

First, we load our model using our `--prototxt`  and `--model`  file paths. We store the model as net.

Then we load the `image` , extract the dimensions, and create a `blob`.

The `dnn.blobFromImage`  takes care of pre-processing which includes setting the `blob`  dimensions and normalization.

Next, we’ll apply face detection:

```
# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
```

To detect faces, we pass the `blob` through the net.
And from there we’ll loop over the `detections`  and draw boxes around the detected faces:

```
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
```

We begin looping over the detections.

From there, we extract the confidence and compare it to the confidence threshold. We perform this check to filter out weak detections.

If the confidence meets the minimum threshold, we proceed to draw a rectangle and along with the *probability* of the detection.

To accomplish this, we first calculate the *(x, y)*-coordinates of the bounding box.

We then build our confidence `text`  string which contains the probability of the detection.

In case the our `text`  would go off-image (such as when the face detection occurs at the very top of an image), we shift it down by 10 pixels.

Our face rectangle and confidence `text`  is drawn on the image.

From there we loop back for additional detections following the process again. If no `detections`  remain, we’re ready to show our output `image`  on the screen.

From there, open up a terminal and execute the following command:

```
$ python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt \
	--model res10_300x300_ssd_iter_140000.caffemodel
```

![face detection](face_detection_01.jpg)

Here you can see face is detected with 74.30% confidence, even though face is at an angle. OpenCV’s Haar cascades are notorious for missing faces that are not at a “straight on” angle, but by using OpenCV’s deep learning face detectors, we are able to detect face.

And now we’ll see how another example works, this time with three faces:

```
$ python detect_faces.py --image iron_chic.jpg --prototxt deploy.prototxt.txt \
	--model res10_300x300_ssd_iter_140000.caffemodel
```

![face detection](face_detection_02.jpg)

Again, this just goes to show how much better (in terms of accuracy) the deep learning OpenCV face detectors are over their standard Haar cascade counterparts shipped with the library.

## Face detection in video and webcam with OpenCV and deep learning
Let’s also apply face detection to videos, video streams, and webcams.

Luckily for us, most of our code in the previous section on face detection with OpenCV in single images can be reused here!

Open up a `detect_faces_video.py` , and see the following code:

```
# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

Compared to above, we will need to import three additional packages: `VideoStream` , `imutils` , and `time` .

If you don’t have imutils  in your virtual environment, you can install it via:

```
$ pip install imutils
```

Our command line arguments are mostly the same, except we do not have an `--image`  path argument this time. We’ll be using our webcam’s video feed instead.

From there we’ll load our model and initialize the video stream:

```
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
```

Loading the model is the same as above.

We initialize a `VideoStream`  object specifying camera with index zero as the source (in general this would be your laptop’s built in camera or your desktop’s first camera detected).

We then allow the camera sensor to warm up for 2 seconds.

From there we loop over the frames and compute face detections with OpenCV:

```
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()
```

This block should look mostly familiar to the static image version.

In this block, we’re reading a `frame`  from the video stream, creating a `blob`  , and passing the blob  through the deep neural `net`  to obtain face detections.

We can now loop over the detections, compare to the confidence threshold, and draw face boxes + confidence values on the screen:

```
# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue
		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
```

For a detailed review of this code block, please review the above section where we perform face detection to still, static images. The code here is nearly identical.

Now that our OpenCV face detections have been drawn, let’s display the frame on the screen and wait for a keypress:

```
# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
```

We display the `frame`  on the screen until the “q” key is pressed at which point we `break`  out of the loop and perform cleanup.

Once you have downloaded the files, running the deep learning OpenCV face detector with a webcam feed is easy with this simple command:

```
$ python detect_faces_video.py --prototxt deploy.prototxt.txt \
	--model res10_300x300_ssd_iter_140000.caffemodel
```