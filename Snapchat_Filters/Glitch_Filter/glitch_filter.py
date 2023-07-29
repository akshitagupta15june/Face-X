import cv2
import numpy as np

# Load the video capture
camera_video = cv2.VideoCapture(0)

while True:
    ret, frame = camera_video.read()
    if not ret:
        break

    # Check for the "ESC" key to exit the loop
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' key to exit
        break

    # Apply random pixel shifting and pixelation to the frame
    shift_amount = 15  # Adjust this value to control the intensity of the glitch effect (reduced from 30)

    # Randomly shift pixels in the red and blue color channels
    dx_red, dy_red = np.random.randint(-shift_amount, shift_amount, 2)
    dx_blue, dy_blue = np.random.randint(-shift_amount, shift_amount, 2)

    # Ensure that shifted images have the same dimensions as the original frame
    frame[:, :, 2] = cv2.warpAffine(frame[:, :, 2], np.float32([[1, 0, dx_red], [0, 1, dy_red]]), frame.shape[:2][::-1])
    frame[:, :, 0] = cv2.warpAffine(frame[:, :, 0], np.float32([[1, 0, dx_blue], [0, 1, dy_blue]]), frame.shape[:2][::-1])

    # Add red and blue glitches to the original frame
    glitched_frame = frame.copy()
    glitched_frame += frame

    # Clip pixel values to stay within [0, 255] range
    glitched_frame = np.clip(glitched_frame, 0, 255)

    # Combine the original frame with the glitched frame
    alpha = 0.5  # Adjust this value to control the intensity of the glitch effect
    frame = cv2.addWeighted(frame, 1 - alpha, glitched_frame, alpha, 0)

    cv2.imshow("Glitch Effect", frame)

# Release the video capture and close all windows
camera_video.release()
cv2.destroyAllWindows()
