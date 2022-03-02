from facealigner.facealigner import FaceAligner
from facealigner.helpers import rect_to_bb, resize
import dlib
import argparse
import cv2
import sys
import traceback


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True, help='path to facial landmark predictor')
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
try:
	predictor = dlib.shape_predictor(args["shape_predictor"])
except:
	traceback.print_exc()
	sys.exit()
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
try:
	image = resize(image, width=800)
except:
	traceback.print_exc()
	sys.exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# show the original input image and detect faces in the grayscale image
cv2.imshow("Input", image)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	faceOrig = resize(image[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)
	# display the output images
	cv2.imshow("Original", faceOrig)
	cv2.imshow("Aligned", faceAligned)
	cv2.waitKey(0)