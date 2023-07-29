import cv2
import dlib
import numpy as np
from math import hypot

camera_video = cv2.VideoCapture(0)
astronaut_mask = cv2.imread("astronaut_filter.png", cv2.IMREAD_UNCHANGED)
star_pattern = cv2.imread("star_pattern.png", cv2.IMREAD_UNCHANGED)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Increase the astronaut scaling factor for a bigger filter
astronaut_scale = 2.4

while True:
    ret, frame = camera_video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Applying astronaut filter on all faces detected by the camera
    for face in faces:
        # Identifying facial landmarks 0, 16, 29
        landmarks = predictor(gray, face)
        left = (landmarks.part(0).x, landmarks.part(0).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)
        middle = (landmarks.part(29).x, landmarks.part(29).y)

        # Calculate the width and height of the astronaut filter
        astronaut_width = int(astronaut_scale * hypot(left[0] - right[0], left[1] - right[1]))
        astronaut_height = int(astronaut_width * astronaut_mask.shape[0] / astronaut_mask.shape[1])

        # Calculate the top-left coordinates for positioning the filter
        top_left = (int(middle[0] - astronaut_width / 2), int(middle[1] - astronaut_height / 2))

        # Ensure the filter does not go out of bounds
        if top_left[0] < 0:
            top_left = (0, top_left[1])
        if top_left[1] < 0:
            top_left = (top_left[0], 0)

        # Calculate the bottom-right coordinates of the filter
        bottom_right = (top_left[0] + astronaut_width, top_left[1] + astronaut_height)

        # Resize the astronaut filter to match the size of the face ROI
        resized_astronaut_mask = cv2.resize(astronaut_mask, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))

        # Extract the region of interest from the frame
        face_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Apply a mask to the astronaut filter
        if astronaut_mask.shape[2] == 4:  # Check if image has an alpha channel
            mask = resized_astronaut_mask[:, :, 3] / 255.0
            mask_inv = 1.0 - mask
            resized_astronaut_mask = resized_astronaut_mask[:, :, :3]  # Remove alpha channel
        else:
            mask = resized_astronaut_mask[:, :, 0] / 255.0
            mask_inv = 1.0 - mask

        # Apply the mask to the region of interest
        for c in range(0, 3):
            face_roi[:, :, c] = (
                mask * resized_astronaut_mask[:, :, c] + mask_inv * face_roi[:, :, c]
            )

        # Update the frame with the modified region of interest
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = face_roi

    # Resize the star pattern to match the size of the frame
    resized_star_pattern = cv2.resize(star_pattern, (frame.shape[1], frame.shape[0]))

    # Use the alpha channel of the star pattern to overlay it on the frame
    mask = resized_star_pattern[:, :, 3] / 255.0
    mask_inv = 1.0 - mask
    resized_star_pattern = resized_star_pattern[:, :, :3]  # Remove alpha channel

    for c in range(0, 3):
        frame[:, :, c] = (
            mask * resized_star_pattern[:, :, c] + mask_inv * frame[:, :, c]
        )

    cv2.imshow("Frame", frame)

    # Breaking the loop if 'ESC' is pressed
    if cv2.waitKey(1) == 27:
        break

# Releasing the VideoCapture Object and closing the windows.
camera_video.release()
cv2.destroyAllWindows()
