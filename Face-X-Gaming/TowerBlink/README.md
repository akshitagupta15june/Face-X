
# TowerBlink

A tower game that can be played by just blinking.
All the user needs to do is blink his/her eyes and a tower is
placed on the tower stack.

# Libaries Used
[OpenCV](https://docs.opencv.org/4.x/)

[CVzone](https://github.com/cvzone/cvzone)

[PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)

[WebBrowser](https://docs.python.org/3/library/webbrowser.html)







## How to run the program

To run this program, run the following commands in terminal

```bash
  pip install -r requirements.txt
```
```bash
  python3 main.py
```
After it launches, click the start button so the game starts.
Blink whenever you want the Tower to drop and try to get the highest score !



## How it works

**Blink Detection**

First the face landmarks are detected. 
Then the 4 points related to eye are obtained.
Then the Horizontal Distance and Vertical Distance between those points are calculated. 
Finally, the ratio between these distance is calculated and if it is
lower than a treshold value, it indicates that the user has blinked.
This value was found by trial and error.

**Playing the Game**

Firstly the game is launched with the help of webbrowser module. Then
once a blink is detected, PyAutoGUI makes a leftclick so the tower is dropped and comes on the Tower Stack.

## Demo

[Click here to watch the demo](https://www.youtube.com/watch?v=zp16Mp4fjWg)


