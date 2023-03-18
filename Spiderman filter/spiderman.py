import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(r".\haarcascade_frontalface_default.xml")

# Load the mask image and resize it
mask = cv2.imread(r".\spider-man--eps--vector-logo.png", cv2.IMREAD_UNCHANGED)
mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Apply the mask to each face
    for (x, y, w, h) in faces:
        # Resize the mask to match the size of the face
        mask_resized = cv2.resize(mask, (w, h))

        # Extract the region of interest (ROI) corresponding to the face
        roi = frame[y:y + h, x:x + w]

        # Split the mask into separate color and alpha channels
        mask_color = mask_resized[:, :, :3]
        mask_alpha = mask_resized[:, :, 3]

        # Invert the alpha channel
        mask_alpha_inv = cv2.bitwise_not(mask_alpha)

        # Apply the mask to the ROI using the alpha channel
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask_alpha_inv)
        masked_mask = cv2.bitwise_and(mask_color, mask_color, mask=mask_alpha)

        # Combine the masked ROI and masked mask
        combined_roi = cv2.add(masked_roi, masked_mask)

        # Replace the ROI with the masked ROI
        frame[y:y + h, x:x + w] = combined_roi

    # Display the resulting frame
    cv2.imshow('Masked Face', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
