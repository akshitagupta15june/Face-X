# Face Biometric using OpenCV
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial_Biometric/Images/Timeattdmain%20(1).png" hight="300px" width="600px" align="left"/>

## 1.Introduction

Face recognition is the technique in which the identity of a human being can be identified using ones individual face. Such kind of systems can be used in photos, videos, or in real time machines. The objective of this article is to provide a simpler and easy method in machine technology. With the help of such a technology one can easily detect the face by the help of dataset in similar matching appearance of a person. The method in which with the help of python and OpenCV in deep learning is the most efficient way to detect the face of the person. This method is useful in many fields such as the military, for security, schools, colleges and universities, airlines, banking, online web applications, gaming etc. this system uses powerful python algorithm through which the detection and recognition of face is very easy and efficien

 This repository detects a human face using Dlib's 68 points model. As the human face is way too complex for a computer to learn, so we have used the 68 points model to ease the process of facial recognition. Facial Biometric uses a two step biometric process for facial recognition. 
 These steps are:
 1. Facial localization to locate a human face and return 4(x,y)-coordinates that forms a rectangle bounding the face.
 2. Detecting facial structures using Dlib's 68 points model. 

## 2.Dlib's 68 points model

 The below image is an example of Dlib's 68 points model. This pre-trained facial landmark detector inside the Dlib's library is used to estimate the location of 68(x,y)-coordinates that maps to the different facial structures. 

###  1.Facial landmark points detection through Dlib's 68 Model:

There are mostly two steps to detect face landmarks in an image which are given below:
- Face detection: Face detection is the first methods which locate a human face and return a value in x,y,w,h which is a rectangle.
- Face landmark: After getting the location of a face in an image, then we have to through points inside of that rectangle.

There are many methods of face detector but we focus in this post only one which is Dlib's method. Like, Opencv uses methods LBP cascades and HAAR and Dlib's use methods HOG `(Histogram of Oriented Gradients)`and SVM `(Support Vector Machine)`.

Now to draw landmarks on the face of the detected rectangle, we are passing the landmarks values and image to the facePoints. In the below code, we are passing landmarks and image as a parameter to a method called drawPoints which accessing the coordinates(x,y) of the ith landmarks points using the part(i).x and part(i).y. All landmarks points are saved in a numpy array and then pass these points to in-built cv2.polylines method to draw the lines on the face using the startpoint and endpoint parameters.

## 3.How to get started

- Clone this repository-
`git clone https://github.com/akshitagupta15june/Face-X.git`
- Change Directory-
`cd Facial_Biometric`

- Run file-
`python library.py`

- Input name-
`Type your name in the input dialogue opened in the terminal`

## Requirements

- python 3.6+
- opencv
- dlib

`Note` : This file takes input video from your webcam and detects the points, So you need an inbuilt or externally connected webcam

## Installation 


- Create virtual environment-
```
- `python -m venv env`
- `source env/bin/activate`  (Linux)
- `pip install opencv-python==4.4.0.44`
- `pip install dlib==19.21.1`
- `pip install opencv-python==4.4.0.44`
```
```
Note : dlib is a library written in c++ that used applications like cmake,boost etc.,if you face any error while installing dlib, don't panic and try to install the extensions required.
```
## Code Overview : 
```
import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
name = input("Enter your name: ")
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),3)
        landmarks = predictor(gray, face)
        # print(landmarks.parts())
        nose = landmarks.parts()[27]
        # print(nose.x, nose.y)
        cv2.putText(frame,str(name),(x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (0, 0, 255), 3)

    # print(faces)

    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

```
## Result Obtain: 



<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial_Biometric/Images/face-b3.png" height="350px" align="left"/>
<p style="clear:both;">
<h1><a name="contributing"></a><a name="community"></a> <a href="https://github.com/akshitagupta15june/Face-X">Community</a> and <a href="https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md">Contributing</a></h1>
<p>Please do! Contributions, updates, <a href="https://github.com/akshitagupta15june/Face-X/issues"></a> and <a href=" ">pull requests</a> are welcome. This project is community-built and welcomes collaboration. Contributors are expected to adhere to the <a href="https://gssoc.girlscript.tech/">GOSSC Code of Conduct</a>.
</p>
<p>
Jump into our <a href="https://discord.com/invite/Jmc97prqjb">Discord</a>! Our projects are community-built and welcome collaboration. 👍Be sure to see the <a href="https://github.com/akshitagupta15june/Face-X/blob/master/Readme.md">Face-X Community Welcome Guide</a> for a tour of resources available to you.
</p>
<p>
<i>Not sure where to start?</i> Grab an open issue with the <a href="https://github.com/akshitagupta15june/Face-X/issues">help-wanted label</a>
</p>



