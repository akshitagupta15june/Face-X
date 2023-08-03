# Air Force Fighter Pilot Flight - Real-time Video Effect Snapchat Filter

This project demonstrates a real-time video effect that superimposes a Fighter Pilot Cockpit Mask on the faces of individuals detected in the webcam feed, while replacing the background with a dynamic sky background obtained from a GIF.

## Implementation Details

### Libraries Used:
- OpenCV: A computer vision library for image and video processing.
- Mediapipe: A library that provides various AI solutions, including Selfie Segmentation for real-time segmentation of foreground and background in images and videos.
- NumPy: A library for numerical computations in Python.
- ImageIO: A library to read and write images and videos.

### Requirements:
- Python 3.x
- OpenCV
- Mediapipe
- NumPy
- ImageIO

### Files:
1. `fighter_pilot_helmet_mask.png`: The Fighter Pilot Cockpit Mask image with an alpha channel (transparency).
2. `bg.gif`: A GIF containing frames of the dynamic sky background.
3. `haarcascade_frontalface_default.xml`: A pre-trained Haar Cascade Classifier for face detection.

### How it Works:

1. Load required libraries and modules.
2. Load the pre-trained Haar Cascade Classifier for face detection from OpenCV.
3. Initialize the Selfie Segmentation model from Mediapipe, which segments the foreground (person) and background in real-time using the webcam feed.
4. Start the video capture from the webcam using OpenCV.
5. Load the dynamic sky background GIF using ImageIO and create a list of frames.
6. Apply Gaussian blur to the first frame of the GIF to create the initial background.
7. While capturing frames from the webcam:
   - Use Selfie Segmentation to obtain the segmentation mask separating the person from the background.
   - Process the segmentation mask to create a condition for selecting the person and background regions.
   - Convert the frame to grayscale for face detection.
   - Detect faces in the grayscale frame using the Haar Cascade Classifier.
   - For each detected face:
     - Calculate the region of interest (ROI) for applying the Fighter Pilot Cockpit Mask.
     - Load the Fighter Pilot Cockpit Mask image and resize it according to the face dimensions.
     - Create masks for the Fighter Pilot Cockpit Mask (alpha channel) and its inverse.
     - Blend the Fighter Pilot Cockpit Mask with the frame ROI to superimpose it on the face.
   - Use the segmentation mask condition to replace the background with the dynamic sky background from the GIF.
   - Update the background with the next frame from the GIF for a dynamic effect.
   - Display the final video with the Fighter Pilot Cockpit Mask and dynamic sky background.
8. The program continues until 'q' is pressed to exit.

### Instructions:

1. Ensure you have Python 3.x and the required libraries installed.
2. Download the `fighter_pilot_helmet_mask.png`, `bg.gif`, and `haarcascade_frontalface_default.xml` files in the same directory as the Python script.
3. Run the Python script.
4. Grant the webcam access when prompted.
5. The webcam feed will open, and the Fighter Pilot Cockpit Mask and dynamic sky background effect will be applied in real-time.

**Note:** For better performance and experience, ensure proper lighting conditions and a clear view of the face while using the application.

### Further Improvements:

- Experiment with different mask images and background animations to create new and exciting effects.
- Optimize the code for better performance, such as by using multi-threading or GPU acceleration.
- Explore other AI models for better segmentation or more advanced effects.

Enjoy the Fighter Pilot Cockpit Mask and Sky Background experience! Feel free to modify the code and unleash your creativity!

## Output screenshot

![Air Force Pilot Filter](https://github.com/Codingpanda252/Face-X/assets/129882142/0f064d73-6fd2-4ca1-acb4-6d5087ead379)

## Author
[Subhasish Panda](https://github.com/Codingpanda252)
