**YOUTUBE CONTROLLER**  

A real-time controller for youtube


**DESCRIPTION**

+ The purpose of this Python script is to identify winks and hand gestures in a real-time live video. 
+ It makes use of the pyautogui package to simulate a key press (space bar in this case) when it detects a wink or a hand gesture.
+ The OpenCV library is used to recognise faces and eyes in each frame that is taken. 
+ Every frame is examined by the script, which continuously runs, to look for winks.
+ When a wink is detected the press of a spacebar is simulated which plays or pauses a video
+ You can use your keyboard to end the programme by pressing the key "q".

**What You Can Do**

You can pause or play a youtube video just by blinking your eyes.

It works with your hands also,you can open and close your hands to pause or play the video

**Libraries Required**

+ OpenCV
+ Numpy
+ pyautogui
+ time module

## How to use the script

* Clone this repository.
```bash
  git clone https://github.com/akshitagupta15june/Face-X.git
```
* Navigate to the required directory.
```bash
  cd Youtube Controller/ytcontroller
```
* Install the Python dependencies.

```bash
  pip install -r requirements.txt
```
* Run the script.
```bash
  python YoutubeController.py
```

## Demostration
Here I have used my hand to control the video

![Snapshot_1](https://user-images.githubusercontent.com/114090255/219082705-06a802c6-d744-46a0-b449-6e0b8690aab1.PNG)

**[Mihir Panchal](https://www.github.com/MihirRajeshPanchal)**

**[B C Samrudh](https://www.github.com/bcsamrudh)**
