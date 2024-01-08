# Awesome Face Operations: Virtual Background
Virtual background involve removing the background
from an image or video stream, and replacing it
with an image/video of our choice.


This is a  real-time filter that creates a virtual
background of our choice on a live feed using using 
Mediapipe's selfie segmentation model.

How it works:

1. The selfie segmentation model is loaded from Mediapipe.

2. Video capture is started from the webcam.

3. The frame size is obtained, and the background image is loaded.

4. A Gaussian blur is applied to the background image.

5. Each frame from the video stream is processed:

   a. The segmentation mask is extracted using selfie segmentation.

   b. A condition mask is created based on the threshold value.

   c. The background is applied to the frame using the mask.

   d. The frames per second (FPS) are counted and displayed.

6. The final result (`output_image`) with the virtual background is displayed using `cv2.imshow`.

7. The loop ends and the windows are closed when 'q' is pressed.


## Sample
Note: The filter was applied to stock footage for the sample gif

![output](output.gif)

## Getting Started

* Clone this repository.
```bash
  git clone https://github.com/akshitagupta15june/Face-X.git
```
* Navigate to the required directory.
```bash
   cd Awesome-face-operations/Virtual_Background
```
* Install the Python dependencies.

```bash
  pip install -r requirements.txt
```
* Run the script.
```bash
  python bg.py
```

Note: Press 'q' to quit the filter
## Author

[Abir-Thakur](https://github.com/Inferno2211)

