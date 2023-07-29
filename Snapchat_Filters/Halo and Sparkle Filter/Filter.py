import cv2
import dlib
import numpy as np
import imageio

# Load the Halo, Light, and Sparkle images
halo_image = cv2.imread('Halo.png', cv2.IMREAD_UNCHANGED)
light_image = cv2.imread('Light.png', cv2.IMREAD_UNCHANGED)
sparkle_gif = imageio.mimread('Sparkle.gif')

# Increase the contrast of the Halo image
halo_contrast = cv2.convertScaleAbs(halo_image, alpha=1.5, beta=0)

# Create the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize the video capture
camera_video = cv2.VideoCapture(0)

# Set the desired output window size
output_width = 800
output_height = 600

# Initialize variables for frame rate and performance optimization
frame_count = 0
skip_frames = 2  # Process every other frame

# Create the output window
cv2.namedWindow('Face Filters', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Filters', output_width, output_height)

while True:
    # Read the frame from the video capture
    ret, frame = camera_video.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames to lower the frame rate
    if frame_count % skip_frames != 0:
        continue

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Resize the frame to a smaller size
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Adjust the scaling factor as needed

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Detect landmarks for the current face
        landmarks = predictor(gray, face)

        # Get the coordinates of the face landmarks
        leftmost_x = min(landmarks.parts(), key=lambda p: p.x).x
        rightmost_x = max(landmarks.parts(), key=lambda p: p.x).x
        topmost_y = min(landmarks.parts(), key=lambda p: p.y).y
        bottommost_y = max(landmarks.parts(), key=lambda p: p.y).y

        # Calculate the width and height of the face
        face_width = rightmost_x - leftmost_x
        face_height = bottommost_y - topmost_y

        # Resize the Halo and Light images to fit the size of the face
        halo_resized = cv2.resize(halo_contrast, (face_width, face_height))
        light_resized = cv2.resize(light_image, (frame.shape[1], frame.shape[0]))

        # Calculate the position to place the Halo image
        x = leftmost_x
        y_halo = topmost_y - int(face_height * 1.2)  # Position the halo above the head

        # Overlay the Halo above the head
        for i in range(face_height):
            for j in range(face_width):
                if halo_resized[i, j, 3] > 0:  # Check the alpha channel for transparency
                    alpha = halo_resized[i, j, 3] / 255.0  # Normalize the alpha value
                    frame_y = y_halo + i
                    frame_x = x + j
                    if frame_y >= 0 and frame_y < frame.shape[0] and frame_x >= 0 and frame_x < frame.shape[1]:
                        frame[frame_y, frame_x] = (1 - alpha) * frame[frame_y, frame_x] + alpha * halo_resized[i, j, :3]

        # Overlay the Light on the frame
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if light_resized[i, j, 3] > 0:  # Check the alpha channel for transparency
                    alpha = light_resized[i, j, 3] / 255.0  # Normalize the alpha value
                    frame[i, j] = (1 - alpha) * frame[i, j] + alpha * light_resized[i, j, :3]

    # Overlay the Sparkle GIF on the frame
    sparkle_frame = sparkle_gif[frame_count % len(sparkle_gif)]
    sparkle_frame_resized = cv2.resize(sparkle_frame[:, :, :3], (frame.shape[1], frame.shape[0]))
    frame = cv2.addWeighted(frame, 1, sparkle_frame_resized, 0.8, 0)

    # Display the resulting frame
    cv2.imshow('Face Filters', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
camera_video.release()
cv2.destroyAllWindows()























