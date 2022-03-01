# BULL NOSE FILTER

## Libraries Used
- OpenCV
- Mediapipe
- Math
- Itertools

## How it's Done:
- Creating facemesh and drwaingspec instance/object.
- Mediapipe detects 468 facial landmarks, I used the nose landmarks.
- Creating the bullnose mask and removing nose area.
- Adding the current frame and bullnose mask.
- Displaying the modified frame

Screenshot of Filter application:
![facemarks points](/nosess.PNG)

