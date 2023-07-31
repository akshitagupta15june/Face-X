# Snapchat filter: Deep Sea
This filter involves detecting the person, 
and replacing the background with a deep sea scene.

It then detects the face and places the scuba helmet,
while allowing the visor to be transparent

This is a  real-time filter that creates an
immersive deep sea scene using HaarCascades
Mediapipe's selfie segmentation model.

How it works:

1. The selfie segmentation model is loaded from Mediapipe.

2. HaarCascades from opencv are loaded

3. Video capture is started from the webcam.

4. The frame size is obtained, and the background image is loaded.

5. Each frame from the video stream is processed:

   a. The segmentation mask is extracted using selfie segmentation.

   b. A condition mask is created based on the threshold value.

   c. The background is applied to the frame using the mask.

   d. The face is detected from the original frame

   e. The scuba helmet is blended on top of the face.

   f. The visor is made semi transperant

6. The final result (`output_image`) with the virtual background is displayed using `cv2.imshow`.

7. The loop ends and the windows are closed when 'q' is pressed.


## Sample

![output](output.png)

## Getting Started

* Clone this repository.
```bash
  git clone https://github.com/akshitagupta15june/Face-X.git
```
* Navigate to the required directory.
```bash
   cd Snapchat filters/Deep\ Sea\ Filter/
```
* Install the Python dependencies.

```bash
  pip install -r requirements.txt
```
* Run the script.
```bash
  python filter.py
```

Note: Press 'q' to quit the filter
## Author

[Abir-Thakur](https://github.com/Inferno2211)

