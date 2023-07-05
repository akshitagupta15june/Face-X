import cv2
import numpy as np

# Load the mask image
mask_image = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the maximum dimension of the face
        face_max_dim = max(w, h)

        # Calculate the scaling factor for the mask based on the face size
        mask_scale_factor = 1.5

        # Calculate the new dimensions for the mask
        mask_width = int(mask_scale_factor * face_max_dim)
        mask_height = int(mask_scale_factor * face_max_dim)

        # Calculate the top-left corner coordinates of the mask region
        mask_x = int(center_x - mask_width / 2)
        mask_y = int(center_y - mask_height / 2)

        # Calculate the bottom-right corner coordinates of the mask region
        mask_end_x = mask_x + mask_width
        mask_end_y = mask_y + mask_height

        # Ensure the mask coordinates are within the frame bounds
        mask_x = max(mask_x, 0)
        mask_y = max(mask_y, 0)
        mask_end_x = min(mask_end_x, frame.shape[1])
        mask_end_y = min(mask_end_y, frame.shape[0])

        # Resize the mask image to match the calculated dimensions
        resized_mask = cv2.resize(mask_image, (mask_end_x - mask_x, mask_end_y - mask_y))

        # Create a mask for the mask image
        mask = resized_mask[:, :, 3] / 255.0

        # Inverse the mask
        inverse_mask = 1.0 - mask

        # Blend the colors of the mask and the frame
        mask_rgb = resized_mask[:, :, :3]
        frame_roi = frame[mask_y:mask_end_y, mask_x:mask_end_x]
        blended_roi = (mask_rgb * mask[:, :, np.newaxis] + frame_roi * inverse_mask[:, :, np.newaxis]).astype(np.uint8)

        # Replace the frame ROI with the blended result
        frame[mask_y:mask_end_y, mask_x:mask_end_x] = blended_roi

    # Display the resulting frame
    cv2.imshow('Mask Overlay', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
