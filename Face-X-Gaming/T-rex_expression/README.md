# Playing Chrome’s T-Rex Game with Facial Gestures
- The only two controls that you need to worry about is making the Dino Jump or Crouch in order to avoid obstacles. Normally, you would press the space and down button on the keyboard to make the Dino do that.So all we need to do is programmatically press those buttons, we can easily do that by utilizing the pyautogui library which will allow us to control our keyboard with python.


![image](https://user-images.githubusercontent.com/67019423/120097033-0aee5980-c14c-11eb-815d-f460fac9ca99.png)

- So all we need to do is programmatically press those buttons, we can easily do that by utilizing the pyautogui library which will allow us to control our keyboard with python.

`pip install PyAutoGUI`

## Requirements
- pyautogui==0.9.52
- matplotlib==3.3.2
- numpy==1.19.2
- opencv-python==4.5.1
- dlib==19.21.1

## Outline:
- Step 1: Real-time Face Detection
- Step 2: Find the landmarks for the detected face
- Step 3: Build the Jump Control mechanism for the Dino
- Step 4: Build the Crouch Control mechanism
- Step 5: Perform Calibration
- Step 6: Keyboard Automation with PyautoGUI
- Step 7: Build the Final Application

## Lets start by importing the required libraries
- `import cv2`
- `import numpy as np`
- `import matplotlib.pyplot as plt`
- `from math import hypot`
- `import pyautogui`
- `import dlib`

## Landmark Detection
![image](https://user-images.githubusercontent.com/67019423/120097056-1fcaed00-c14c-11eb-8ed9-3c838a72c688.png)

## Jump Control mechanism
![image](https://user-images.githubusercontent.com/67019423/120097075-38d39e00-c14c-11eb-8b01-8be5ff9d1db9.png)

## Crouch Control Mechanism
![image](https://user-images.githubusercontent.com/67019423/120097325-78e75080-c14d-11eb-8f1c-acd032dbd7dd.png)

## Final Application
![ezgif com-gif-maker](https://user-images.githubusercontent.com/67019423/120097166-9f58bc00-c14c-11eb-83f0-1ff1b7cbb48f.gif)
