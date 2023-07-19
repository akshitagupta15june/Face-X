# <b> Funny Shrek Snapchat Filter </b>

## Setting up and running the script

- Install the requirements of the script as follows
```
    $ pip install -r requirements.txt
```

- Run the script as follows
```
    $ python funny_shrek.py
```

# Real-Time Funny Shrek Filter using OpenCV and Dlib

This project is an implementation of a real-time Funny Shrek filter using OpenCV and Dlib. It applies a Funny Shrek mask overlay to the faces detected in a live video stream from a webcam.

# Description 

1. Importing the necessary libraries:
   - `cv2`: OpenCV library for computer vision tasks.
   - `dlib`: Library for face detection and facial landmark detection.

2. Defining the `filter` function:
   - This function takes the current frame and facial landmarks as input.
   - It loads the Funny Shrek filter image and its corresponding binary mask.
   - The Funny Shrek filter image is resized to match the size of the region of interest (ROI) on the face.
   - The ROI is extracted from the frame using the facial landmarks.
   - The background and foreground regions of the ROI are separated using the binary mask.
   - The Funny Shrek filter is applied to the ROI by combining the background and foreground regions.
   - The modified ROI is placed back into the frame.
   - The modified frame is returned.

3. Initializing the face detector and facial landmark predictor:
   - The `dlib.get_frontal_face_detector()` function is used to create a face detector object.
   - The `dlib.shape_predictor()` function is used to load the pre-trained facial landmark predictor.

4. Capturing the video stream from the webcam:
   - The `cv2.VideoCapture()` function is used to create a video capture object.
   - Inside the while loop, frames are continuously read from the video capture object.

5. Preprocessing the frame:
   - Each frame is converted to grayscale using the `cv2.cvtColor()` function.

6. Detecting faces in the frame:
   - The face detector is applied to the grayscale frame, returning a list of detected face rectangles.

7. Applying the Funny Shrek filter to each detected face:
   - For each detected face rectangle, facial landmarks are determined using the facial landmark predictor.
   - The `filter` function is called to apply the Funny Shrek filter to the current frame using the facial landmarks.

8. Displaying the modified frame:
   - The modified frame, with the Funny Shrek filter applied to the detected faces, is displayed using the `cv2.imshow()` function.

9. Exiting the program:
   - The program continues to run until the 'Esc' key is pressed.
   - When the 'Esc' key is pressed, the video capture is released and all windows are closed using the `cv2.waitKey()` and `cv2.destroyAllWindows()` functions.



Note: Ensure that your system has a webcam connected and properly configured.

## Output screenshot

![Funny_Shrek](https://github.com/Codingpanda252/Face-X/assets/129882142/ef852da4-ce11-4155-8d4a-73eb3dbd33e7)


## Author
[Subhasish Panda](https://github.com/Codingpanda252)
