import cv2
import mediapipe as mp
import numpy as np
import imageio

print("All modules imported!")
# Set variables
threshold = 0.5
b_amt = 15

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

print("Starting video capture...")
# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Find out frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load background GIF using imageio.get_reader()
gif_path = "bg.gif"
gif_reader = imageio.get_reader(gif_path)
gif_frames = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),(width,height)) for frame in gif_reader]
gif_frame_idx = 0
# Apply gaussian blur to initial background
bg_image = cv2.GaussianBlur(gif_frames[0], (b_amt, b_amt), 0)

print("Capture started!")
while True:
    # Read frame from video stream
    ret, frame = cap.read()

    # Create a mask for person and background
    results = selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    segmentation_mask = results.segmentation_mask

    # Process mask using threshold value
    condition = np.stack((segmentation_mask,) * 3, axis=-1) > threshold

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Use the mask to apply background to frame
    frame_with_background = np.where(condition, frame, bg_image)

    # Update background to the next frame in the GIF
    gif_frame_idx = (gif_frame_idx + 1) % len(gif_frames)
    bg_image = cv2.GaussianBlur(gif_frames[gif_frame_idx], (b_amt, b_amt), 0)
    
    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the maximum dimension of the face
        face_max_dim = max(w, h)

        # Calculate the scaling factor for the mask based on the face size
        mask_scale_factor = 2.5

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

        # Load the mask image
        mask_image = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)

        # Resize the mask image to match the calculated dimensions
        resized_mask = cv2.resize(mask_image, (mask_end_x - mask_x, mask_end_y - mask_y))

        # Create a mask for the mask image
        mask = resized_mask[:, :, 3] / 255.0

        # Inverse the mask
        inverse_mask = 1.0 - mask

        # Blend the colors of the mask and the frame_roi
        mask_rgb = resized_mask[:, :, :3]
        frame_roi = frame_with_background[mask_y:mask_end_y, mask_x:mask_end_x]
        blended_roi = (mask_rgb * mask[:, :, np.newaxis] + frame_roi * inverse_mask[:, :, np.newaxis]).astype(np.uint8)

        # Replace the frame_roi with the blended result
        frame_with_background[mask_y:mask_end_y, mask_x:mask_end_x] = blended_roi

    # Display final result
    cv2.imshow("Virtual background", frame_with_background)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
