# importing necessary libraries

import cv2
import dlib
import argparse

import numpy as np

from imutils import resize as ReSize
from collections import OrderedDict

"""The pre-trained facial landmark detector inside the dlib library is 
 used to estimate the location of 68 (x, y)-x_y that map to facial 
 structures on the face. The 68 landmark output is shown in the figure below."""


# defining the 2 main functions 
# convert_shape_to_array -> to obtain the x, y coordinates in an array
# highlight_features -> to locate and visualize the features separately with colours

def convert_shape_to_array(face_feature, dtype="int"):

    # initializing
    x_y = np.zeros((68, 2), dtype=dtype)
    for index in range(0, 68):
        # Converting each facial indexes to x, y coordinates by looping

        x_y[index] = (face_feature.part(index).x, face_feature.part(index).y)
    return x_y

def highlight_features(image, feature, colors=None, alpha=0.75):
    # making copies of input image

    img_overlay = image.copy()
    output_img = image.copy()

    # different colours for different landmarks if colours is None
    if colors is None:
        colors = [(158, 163, 32), (230, 159, 23), (19, 199, 109), (180, 42, 220), (168, 100, 168), (79, 76, 240), (163, 38, 32),  (158, 163, 32)]

    # Looping to find jawline because jawline is non enclosed facial region

    for (ind, landmark) in enumerate(landmark_index.keys()):
        (start_ind, end_ind) = landmark_index[landmark]
        feature_ind = feature[start_ind: end_ind]

        feature_x_y[ind] = feature_ind
        # check if are supposed to draw the jawline
        if landmark == "Jaw":
            # outlining from point x to y
            for l in range(1, len(feature_ind)):
                jaw_start = tuple(feature_ind[l - 1])
                jaw_end = tuple(feature_ind[l])
                cv2.line(img_overlay, jaw_start, jaw_end, colors[ind], 2)

        # if not, highlight the feature after computing connvex hull by drawing contours
        else:
            hull = cv2.convexHull(feature_ind)

            # contours to make it visually highlighted and addweighted to plly transparent overlay in the end
            cv2.drawContours(img_overlay, [hull], -1, colors[ind], -1)
    cv2.addWeighted(img_overlay, alpha, output_img, 1 - alpha, 0, output_img)
    # return the output image
    return output_img

# creating a dictionary to store the landmarks of each feature from the top
# features include jaw, right left eyebrows, nose, right left eyes and mouth

landmark_index = OrderedDict([("Jaw", (0, 17)), ("Right_Eyebrow", (17, 22)), ("Left_Eyebrow", (22, 27)), ("Nose", (27, 35)), ("Right_Eye", (36, 42)), ("Left_Eye", (42, 48)), ("Mouth", (48, 68))])

facial_landmarks = dlib.get_frontal_face_detector()

face_feature_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
feature_x_y = {}

# use ArguementParser to create arguement lines to obtain input image

arg_pars = argparse.ArgumentParser()

#obtaining input image
arg_pars.add_argument("-i", "--image", required=True, help="path to input image")

args = vars(arg_pars.parse_args())

# openCV to load and resize the image
input_image = ReSize(cv2.imread(args["image"]), width=800)

# convert input image to grayscale for further processing
grayscale_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Predicting the face of the image and looping
face_detec = facial_landmarks(grayscale_img, 1)
for (landmrk_ind, face_detec) in enumerate(face_detec):
    # obtaining face region landmarks and converting coordinates (x, y) to array
    output_img = highlight_features(input_image, convert_shape_to_array(face_feature_landmarks(grayscale_img, face_detec)))

    # display image
    cv2.imshow("Face Features", output_img)
    cv2.waitKey(0)
