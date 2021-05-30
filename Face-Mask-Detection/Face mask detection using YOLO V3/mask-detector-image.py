# import the necessary libs
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-o", "--output",help="path to output image")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.45,help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applying non-max suppression")
args = vars(ap.parse_args())

# load the class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label (red and green)
COLORS = [[0,0,255],[0,255,0]]

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov4_face_mask.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4-obj.cfg"])

# load our YOLO object detector 
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and get it height and width
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()

ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (832, 832),swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln) #list of 3 arrays, for each output layer.
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:

	# loop over each of the detections
	for detection in output:

		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:] #last 2 values in vector
		classID = np.argmax(scores)
		confidence = scores[classID]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply NMS to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])
border_size=100
border_text_color=[255,255,255]
#Add top-border to image to display stats
image = cv2.copyMakeBorder(image, border_size,0,0,0, cv2.BORDER_CONSTANT)
#calculate count values
filtered_classids=np.take(classIDs,idxs)
mask_count=(filtered_classids==1).sum()
nomask_count=(filtered_classids==0).sum()
#display count
text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
cv2.putText(image,text, (0, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,border_text_color, 2)
#display status
text = "Status:"
cv2.putText(image,text, (W-300, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,border_text_color, 2)
ratio=nomask_count/(mask_count+nomask_count)

if ratio>=0.1 and nomask_count>=3:
	text = "Danger !"
	cv2.putText(image,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,[26,13,247], 2)
	
elif ratio!=0 and np.isnan(ratio)!=True:
	text = "Warning !"
	cv2.putText(image,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,[0,255,255], 2)

else:
	text = "Safe "
	cv2.putText(image,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,[0,255,0], 2)

# ensure at least one detection exists
if len(idxs) > 0:

	# loop over the indexes we are keeping
	for i in idxs.flatten():
		
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1]+border_size)
		(w, h) = (boxes[i][2], boxes[i][3])
		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

if args["output"]:
	#save the image
	cv2.imwrite(args["output"],image)		 
# show the output image
cv2.imshow("Image",image)
cv2.waitKey(0)