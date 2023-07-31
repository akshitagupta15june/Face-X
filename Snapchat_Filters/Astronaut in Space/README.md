# Astronaut in Space Snapchat filter using OpenCV and dlib
A filter with a video feed with the astronaut filter on faces and the star pattern on the entire frame using OpenCV and dlib python libraries. 

# Working of Astronaut in Space filter:
The filter in this code is a combination of two PNG images that are applied to the live camera feed:

1. Astronaut Filter:
   - The astronaut filter is an image of an astronaut helmet with a transparent background. It is loaded from the PNG file "astronaut_filter.png".
   - The filter is applied to each detected face in the camera feed. To do this, the code uses the dlib library to detect facial landmarks, specifically the left eye, right eye, and the middle of the nose bridge (landmarks 0, 16, and 29).
   - Based on the positions of these facial landmarks, the width and height of the astronaut filter are calculated using a scaling factor to make it larger. The top-left and bottom-right coordinates are determined to position the filter correctly on the face.
   - The astronaut filter is then resized to match the size of the face region of interest (ROI) and a mask is created to control the transparency of the filter using the alpha channel (if available).
   - The filter is blended with the face ROI using the mask and mask inversion, which allows the camera feed to be visible through the transparent parts of the filter.
   - The result is a modified frame with the astronaut filter applied to each detected face.

2. Star Pattern:
   - The star pattern is another PNG image with a transparent background. It is loaded from the file "star_pattern.png".
   - The star pattern is resized to match the size of the entire frame (the camera feed).
   - Similar to the astronaut filter, a mask is created to control the transparency of the star pattern using the alpha channel (if available).
   - The star pattern is blended with the entire frame using the mask and mask inversion, allowing the live camera feed to show through the transparent parts of the pattern.
   - The result is a modified frame with the star pattern overlaid on top of the entire camera feed.

# Implementation of Astronaut in Space filter:
   - The code captures frames from the live camera feed using OpenCV's `VideoCapture` class.
   - It uses dlib's face detector and facial landmark predictor to identify faces and specific facial landmarks in each frame.
   - For each detected face, the astronaut filter is applied using the calculated scaling factor and facial landmarks to determine the position and size.
   - The star pattern is then applied to the entire frame by resizing it to match the frame's size and blending it with the frame using the mask technique.
   - The modified frame with the astronaut filter on faces and the star pattern on the entire frame is displayed in real-time using `cv2.imshow`.
   - The loop continues capturing frames until the 'ESC' key is pressed, at which point the program releases the camera feed and closes all windows.

Overall, the implementation uses image processing techniques and blending operations to overlay the PNG filters on the live camera feed, resulting in the desired visual effects of the astronaut filter on faces and the star pattern covering the entire frame.

## Output screenshot

![Astronaut_filter](https://github.com/Codingpanda252/Face-X/assets/129882142/e6ea1135-3864-477b-a433-f03545d34c40)

## Author
[Subhasish Panda](https://github.com/Codingpanda252)
