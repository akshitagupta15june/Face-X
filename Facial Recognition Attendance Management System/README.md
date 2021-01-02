
## About:🤔💭
- Facial recognition attendance management system. It makes use of openCV and python to recognise face. 

- User has to:

  - Click on 'Automatic attendance'.
  - Enter the subject name.
  - Click on 'Fill attendance' and wait for camera window to open.
  - It will recognise the face and display enrollment number and name.
  - Click on 'Fill attendace'.

----

### List TO-DO📄:

- [x] Check whether your system has a web cam or not. If not, then get one camera for face recognisation.
- [x] Install [Python](https://www.howtogeek.com/197947/how-to-install-python-on-windows/)
- [x] Install [Dependencies.](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#dependencies)
- [x] Make a folder and name it anything(or you can see [quick start](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#quick-start))
    - [x] Download [Haarcascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and paste it here
    - [x] Open notepad,write [this](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/face.py) script and save it as 'face.py'.
    - [x] Paste [arduino code](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/servo.ino) in [Arduino IDE](https://www.arduino.cc/en/guide/windows) and save it as 'servo.ino'.
- [x] Assemble [rotation platform](https://www.learnrobotics.org/blog/how-to-assemble-pan-tilt-for-arduino-servos/)
- [x] Make [connections.](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#connections)
- [x] [Test](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#testing) the code.
- [x] Fit the camera on rotation platform.
- [x] Run the final project.

----

### Requirements:🧱🧱

|Hardware|Software|
|----|-----|
|web cam or camera|python|

----

### Dependencies🔧🛠:
Open terminal and write:

* `pip install Pillow`
* `pip install opencv-python`
* `pip install pandas`


## Quick Start📘
You can directly [download](https://www.wikihow.com/Download-a-GitHub-Folder) the entire [Face-X](https://github.com/akshitagupta15june/Face-X) and select the folder you want. All you have to do is now assemble hardware part.


## Connections🔗:

![ ](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/connection%201.png)
![ ](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/connection%202.png)

## Testing🧰:

- After everything is done last thing to do is test if it works.   
- To test first make sure that servos are properly connected to arduino and sketch is uploaded.
- After sketch is uploaded make sure to close the IDE so the port is free to connect to python.
- Now open 'face.py' with Python IDLE and press 'F5' to run the code. It will take a few seconds to connect to arduino and then you should be able to see it working.
- The camera will move in the same direction as of the face since the code is trying to detect a face in the environment.
>For better understanding watch [this](https://www.youtube.com/watch?v=O3_C-R7Jrvo)
