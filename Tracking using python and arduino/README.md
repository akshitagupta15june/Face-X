
## About:🤔💭
Face tracking using python and arduino. An Arduino UNO is programmed using python language which makes the camera moves in the direction of face present.

### List TO-DO📄:

- [x] Get the [hardware.](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#requirements)
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

### Requirements:🧱🧱

|Hardware|Software|
|----|-----|
|[Arduino UNO](https://www.banggood.in/Wholesale-Geekcreit-UNO-R3-ATmega16U2-AVR-USB-Development-Main-Board-Geekcreit-for-Arduino-products-that-work-with-official-Arduino-boards-p-68537.html?akmClientCountry=IN&p=1L111111347088201706&cur_warehouse=CN)|[Python 2.7 or newer](https://www.howtogeek.com/197947/how-to-install-python-on-windows/)|
|Web Cam or phone camera|[Haarcascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)|
|[2 x 9g servos](https://www.banggood.in/6PCS-SG92R-Micro-Digital-Servo-9g-2_5kg-For-RC-Airplane-p-1164389.html?p=1L111111347088201706&custlinkid=796242&cur_warehouse=CN)||
|[Breadboard](https://www.banggood.in/Geekcreit-MB-102-MB102-Solderless-Breadboard-+-Power-Supply-+-Jumper-Cable-Kits-p-933600.html?cur_warehouse=CN&rmmds=search)||
|[Servo Pan Tilt Kit](https://www.banggood.in/Two-DOF-Robot-PTZ-FPV-Dedicated-Nylon-PTZ-Kit-With-Two-9G-Precision-160-Degree-Servo-p-1063479.html?p=1L111111347088201706&cur_warehouse=CN)||


### Dependencies🔧🛠:
Open terminal and write:
* `pip install numpy`
* `pip install serial`
* `pip install opencv-python`


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
- The camera will move in the direction where it will find a face.
>For better understanding watch [this](https://www.youtube.com/watch?v=O3_C-R7Jrvo)
