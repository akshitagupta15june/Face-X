# Face Detect-Recog using Raspberry Pi and OpenCV🎭

## About:🤔💭

This project uses python and OpenCV to recognize multiple faces and show the name #Sample to get video from PiCam.

## List TO-DO📄:

- [x] Get the [hardware.](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#requirements)
- [x] Install [Python](https://www.howtogeek.com/197947/how-to-install-python-on-windows/)
- [x] Install [Dependencies.](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#dependencies)
- [x] Make a folder and name it anything(or you can see [quick start](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#quick-start))
    - [x] Download [Haarcascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and paste it here
    - [x] Open notepad,write [this](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/face.py) script and save it as 'face.py'.
    - [x] Paste [arduino code](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/servo.ino) in [Arduino IDE](https://www.arduino.cc/en/guide/windows) and save it as 'servo.ino'.
- [x] [Test](https://github.com/smriti1313/Face-X/blob/master/Tracking%20using%20python%20and%20arduino/README.md#testing) the code.
- [x] Run the final project.

### Requirements:🧱🧱

|Hardware|Software|
|----|-----|
|[Raspberry PiCam](https://www.raspberrypi.org/products/camera-module-v2/) or Web cam|[Python 2.7 or newer](https://www.howtogeek.com/197947/how-to-install-python-on-windows/)|
|Web Cam or phone camera|[Haarcascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)|



### Dependencies🔧🛠:
Open terminal and write:
* `sudo pip install picamera`
* `sudo pip install "picamera[array]"`
* `pip install opencv-python`


## Quick Start📘
You can directly [download](https://www.wikihow.com/Download-a-GitHub-Folder) the entire [Face-X](https://github.com/akshitagupta15june/Face-X) and select the folder you want. All you have to do is now assemble hardware part.

## Initialize the camera and grab a reference to the raw camera capture

camera = PiCamera() camera.resolution = (640, 480) 

camera.framerate = 32 

rawCapture = PiRGBArray(camera, size=(640, 480))

## Allow the camera to warmup

time.sleep(0.1)

## capture frames from the camera
```py
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True): 
    # grab the raw NumPy array representing the image, then initialize the timestamp and occupied/unoccupied text image = frame.array

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    #clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
```
## Testing🧰:

- Run the code and observe if it working fine or not.

>For better understanding watch [this](https://www.youtube.com/watch?v=Fggavxx-Kds)
