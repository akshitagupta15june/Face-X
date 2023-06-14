import cv2
import dlib
import numpy as np
from math import hypot

import cv2
import dlib
import numpy as np
from math import hypot

camera_video = cv2.VideoCapture(0)
shrek_mask = cv2.imread("funny_shrek.png", cv2.IMREAD_UNCHANGED)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = camera_video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Applying Shrek filter on all faces detected by the camera
    for face in faces:
        # Identifying facial landmarks 0, 16, 29
        landmarks = predictor(gray, face)
        left = (landmarks.part(0).x, landmarks.part(0).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)
        middle = (landmarks.part(29).x, landmarks.part(29).y)

        # Calculate the width and height of the Shrek filter with an increased scaling factor
        shrek_scale = 2.0  # Adjust the scale to make the filter bigger
        shrek_width = int(shrek_scale * hypot(left[0] - right[0], left[1] - right[1]))
        shrek_height = int(shrek_width * shrek_mask.shape[0] / shrek_mask.shape[1])

        # Calculate the top-left coordinates for positioning the filter
        top_left = (int(middle[0] - shrek_width / 2), int(middle[1] - shrek_height / 2))

        # Ensure the filter does not go out of bounds
        if top_left[0] < 0:
            top_left = (0, top_left[1])
        if top_left[1] < 0:
            top_left = (top_left[0], 0)

        # Calculate the bottom-right coordinates of the filter
        bottom_right = (top_left[0] + shrek_width, top_left[1] + shrek_height)

        # Resize the Shrek filter to match the size of the face ROI
        resized_shrek_mask = cv2.resize(shrek_mask, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))

        # Extract the region of interest from the frame
        face_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Apply a mask to the Shrek filter
        if shrek_mask.shape[2] == 4:  # Check if image has an alpha channel
            mask = resized_shrek_mask[:, :, 3] / 255.0
            mask_inv = 1.0 - mask
            resized_shrek_mask = resized_shrek_mask[:, :, :3]  # Remove alpha channel
        else:
            mask = resized_shrek_mask[:, :, 0] / 255.0
            mask_inv = 1.0 - mask

        # Apply the mask to the region of interest
        for c in range(0, 3):
            face_roi[:, :, c] = (
                mask * resized_shrek_mask[:, :, c] + mask_inv * face_roi[:, :, c]
            )

        # Update the frame with the modified region of interest
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = face_roi

    cv2.imshow("Frame", frame)

    # Breaking the loop if 'ESC' is pressed
    if cv2.waitKey(1) == 27:
        break

# Releasing the VideoCapture Object and closing the windows.
camera_video.release()
cv2.destroyAllWindows()
camera_video = cv2.VideoCapture(0)
shrek_mask = cv2.imread("funny_shrek.png", cv2.IMREAD_UNCHANGED)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = camera_video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Applying Shrek filter on all faces detected by the camera
    for face in faces:
        # Identifying facial landmarks 0, 16, 29
        landmarks = predictor(gray, face)
        left = (landmarks.part(0).x, landmarks.part(0).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)
        middle = (landmarks.part(29).x, landmarks.part(29).y)

        # Calculate the width and height of the Shrek filter with an increased scaling factor
        shrek_scale = 2.0  # Adjust the scale to make the filter bigger
        shrek_width = int(shrek_scale * hypot(left[0] - right[0], left[1] - right[1]))
        shrek_height = int(shrek_width * shrek_mask.shape[0] / shrek_mask.shape[1])

        # Calculate the top-left coordinates for positioning the filter
        top_left = (int(middle[0] - shrek_width / 2), int(middle[1] - shrek_height / 2))

        # Ensure the filter does not go out of bounds
        if top_left[0] < 0:
            top_left = (0, top_left[1])
        if top_left[1] < 0:
            top_left = (top_left[0], 0)

        # Calculate the bottom-right coordinates of the filter
        bottom_right = (top_left[0] + shrek_width, top_left[1] + shrek_height)

        # Resize the Shrek filter to match the size of the face ROI
        resized_shrek_mask = cv2.resize(shrek_mask, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))

        # Extract the region of interest from the frame
        face_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Apply a mask to the Shrek filter
        if shrek_mask.shape[2] == 4:  # Check if image has an alpha channel
            mask = resized_shrek_mask[:, :, 3] / 255.0
            mask_inv = 1.0 - mask
            resized_shrek_mask = resized_shrek_mask[:, :, :3]  # Remove alpha channel
        else:
            mask = resized_shrek_mask[:, :, 0] / 255.0
            mask_inv = 1.0 - mask

        # Apply the mask to the region of interest
        for c in range(0, 3):
            face_roi[:, :, c] = (
                mask * resized_shrek_mask[:, :, c] + mask_inv * face_roi[:, :, c]
            )

        # Update the frame with the modified region of interest
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = face_roi

    cv2.imshow("Frame", frame)

    # Breaking the loop if 'ESC' is pressed
    if cv2.waitKey(1) == 27:
        break

# Releasing the VideoCapture Object and closing the windows.
camera_video.release()
cv2.destroyAllWindows()
