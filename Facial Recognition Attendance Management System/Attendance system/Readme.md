<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/Readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/images/932_generated.png"  width="100%" /></a><br /><br /></p>

# Facial Recognition Attendance Management System 

## INTRODUCTION
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/images/FS.png" height="400px" align="right"/>
A facial recognition system is a computer application
which is capable of identifying or verifying a person from a
digital image or a video frame from a video source. One of
the ways to do this is by comparing selected facial features
from the image and a facial database Face recognition
system have improved dramatically in their performance
over the past few years, and this technology is now widely
used for various purposes such as for security and for
commercial applications.Face recognition is an active area
of research which is a computer based digital technology.
Use of face recognition for the purpose of attendance
marking is a smart way of attendance system

It is typically
used in security systems and can be compared to other
biometrics such as fingerprint or eye iris recognition
systems. Recently, it has also become popular as a
commercial identification and marketing tool.As the
number of students in a college or employees at an
organization are increasing the requirements for lecturers or
to the organization is increase the complexity of attendance
monitoring and number of persons present.

## OVERVIEW
Face  recognition  being  a  biometric  technique  implies determination if the image of the face of any particular person matches any of the face images that are stored in a database. This difficulty is tough to resolve automatically because of the changes that several factors, like facial expression, aging and even lighting can affect the image. Facial recognition among the various biometric techniques may not be the most authentic but it has various advantages over the others. Face recognition is  natural,  feasible  and  does  not  require  assistance.  The expected system engages the face recognition approach for the automating the attendance procedure of students or employees without their involvement. A web cam is used for capturing the images of students or employees. The faces in the captured images are detected and compared with the images in database and the attendance is marked. 

## IMAGE PROCESSING 
The facial recognition process can be split into two major stages: processing which occurs before detection involving face detection and alignment and later recognition is done using feature extraction and matching steps.
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/images/FRA4.PNG" align="right"/>
 - `FACE DETECTION `: 
    The primary function of this step is to conclude whether the human faces emerge in a given image, and what is the location of these faces. The expected outputs of this step     are patches which contain each face in the input image. In order to get a more  robust and  easily designable  face recognition  system Face  alignment  is  performed  to       rationalise  the  scales  and orientation of these patches.
 - `FEATURE EXTRACTION` : 
     Following the face detection step the extraction of human face patches from images is done. After this step, the conversion of face patch is done into vector with fixed        coordinates or a set of landmark  points. 
-  `FACE RECOGNITION` :
   The last step  after the  representation of  faces is  to identify them.  For  automatic  recognition  we  need  to  build  a  face database. Various images are taken foe      each person and their features are extracted and stored in the database.  Then when an input image is fed the face detection and feature extraction is performed and its        feature to each face class is compared and stored in the database.
   
## ALGORITHM  
  There  are  various  algorithms  used  for  facial  recognition. Some of them are as follows: 
   1. `Eigen faces`
   2. `Fisher faces`
   3. ` Local binary patterns histograms`

### 1. EIGEN FACES 
This  method is  a statistical  plan. The  characteristic  which influences the images is derived by this algorithm. The whole recognition method will depend on the training database that will be provided. The images from two different classes are not treated individually.
### 2. FISHER FACES 
Fisher faces  algorithm also follows a progressive  approach just like the Eigen faces. This method is a alteration of Eigen faces so it uses the same principal Components Analysis. The major conversion is that the fisher faces considers the classes. As  mentioned  previously,  the  Eigen  faces  does  not differentiate  between  the  two  pictures  from  two  differed classes while training. The total average affects each picture. A  Fisher  face  employs  Linear  Discriminant  Analysis  for distinguishing between pictures from a different class.
### 3.LOCAL BINARY PATTERNS HISTOGRAMS 
This method needs the gray scale pictures for dealing with the training  part.  This  algorithm  in  comparison  to  other algorithms is not a `holistic approach`.  A. PARAMETERS: `LBPH` uses the following parameters: 
-   `Radius: ` Generally  1 is  set as  a radius for  the circular  local binary pattern which denotes the radius around the central pixel.
- ` Neighbours:` The number of  sample  points surrounding the central pixel which is generally 8.The computational cost will increase with increase in number of sample points.
-  ` Grid X: ` The  number  of  cells  along  the  horizontal  direction  is represented as Grid X. With the increase in number of cells the grid becomes finer which results in       increase of dimensional feature vector. 
-  ` Grid Y`: The number of cells along the vertical direction is represented as  Grid  Y. With  the  increase in  number  of  cells  the grid becomes finer which results in increase of dimensional feature vector.

## ALGORITHM TRAINING: 
For the training purpose of the dataset of the facial images of the  people  to be    recognized along  with  the unique  ID  is required  so  that  the  presented  approach  will  utilize  the provided  information  for  perceiving  an  input  image  and providing the output. Same images require same ID. 

## COMPUTATION OF THE ALGORITHM: 
The intermediate  image with improved facial  characteristics which corresponds to the original image is created in the first step. Based on the parameters provided, sliding window theory is used in order to achieve so. Facial  image  is  converted  into  gray  scale.  A  3x3  pixels window is taken which can also be expressed as a 3x3 matrix which contains the intensity of each pixel (0-255). After this we consider the central value of the matrix which we take as the threshold. This value defines the new values obtained from the 8 neighbours. A new binary value is set for each neighbour of the central value. For the values equal to or greater than the threshold value 1 will be the output otherwise 0 will be  the output. Only binary values will be present in the matrix and the  concatenation is  performed at  each position  to get  new values at each position.  Then the conversion  of this binary value into a decimal value is done which is made the central value of the matrix. It is a pixel of the actual image. As the process is completed, we get a new image which serves as the better characteristics of the original image.
## EXTRACTION OF HISTOGRAM:
The image obtained in the previous step uses the Grid X and Grid Y parameters and the image is split into multiple grids. Based on the image the histogram can be extracted as below: 1. The image is in gray scale and each histogram will consist of only 256 positions (0-255) which symbolises the existences of each pixel intensity. 2. After this each histogram is created and a new and bigger histogram is done. Let us suppose that there are 8x8 grids, then there will be 16.384 positions in total in the final histogram. Ultimately the  histogram signifies the features of the actual image. 
## THE FACE RECOGNITION:
The training of the algorithm is done.  For finding the image which  is same  as  the input  image,  the two  histograms  are compared  and  the  image  corresponding  to  the  nearest histogram is returned.   Different approaches are used for the calculation of distance between the two histograms. Here we use the Euclidean distance based on the formula

## ADVANTAGES OF USING LBPH ALGORITHM: 
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/images/images.jpg" align="right"/>

1. It is one of the simplest algorithms for face recognition.
2. The local features of the images can be characterized by this algorithm.
3. Using this algorithm, considerable results can be obtained. 
4. Open CV library is used to implement LBPH algorithm

## DATABASE CREATION: 
The first step in the Attendance  System is  the creation  of a database of faces that will be used. Different individuals are considered and a camera is used for the detection of faces and the recording of the frontal face. The number of frame to be taken for consideration can be modified for accuracy levels. These images are then stored in the database along with the Registration ID. 

## TRAINING OF FACES:
The images are saved in gray scale after being recorded by a camera.  The LBPH  recognizer  is employed  to coach  these faces because the coaching sets the resolution and therefore the recognized face resolutions are completely variant. A part of the  image  is  taken  as  the  centre  and  the  neighbours  are thresholded  against it.  If  the  intensity of  the  centre part  is greater or equal than it neighbour then it is denoted as 1 and 0 if not. This will result in binary patterns generally known as LBP code.

Colons can be used to align columns.

| ACE           |  FACE         |     |
| ------------- |:-------------:| -----:|
| 1.Confidence factor based on output is 2,000-3,000.  | 	It is 100-400                           |          -                    |
| 2.Threshold value is 4,000.                          | Threshold value is 400                   | Threshold value is 7.         |
| 3.Principle of dataset generation is component based | It is component based.                   | It is pixel based.            |
| 4.Basic principle is PCA.                            | Basic principle is LDA.                  | Basic principle is Histogram. |
| 5.Background noise is maximum.                       | Background noise is medium.              | Background noise is minimum.  |
| 6.Efficiency is minimum.                             | Efficiency is greater than Eigen face.   | Efficiency is maximum.        |

## FACE DETECTION:
The data of the trained faces is stored in .py format. The faces are detected using the Haar cascade frontal face module.
## FACE RECOGNITION: 
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/images/Flowchart-for-real-time-face-detection-and-recognition.png" height="50%" width="40%" align="right" />
The data of the trained faces are stored and the detected faces are compared to the IDs of the students and recognized. The recording  of  faces  is  done  in  real  time  to  guarantee  the accuracy of the system. This system is precisely dependant on the camera’s condition.

The training process starts with traversing of the training data directory. Each image  in the training  date is converted into gray scale. A part of the image is taken as center and threshold its neighbours against it. If the intensity of the middle part is more or equal than its neighbour then denote it with 1 and 0 if not. After this the  images are  resized. Then the images  are converted  into  a  numpy  array  which  is  the  central  data structure  of  the  numpy  library. Each  face  in  the image  is detected. Creation of separate lists of each face is done and the faces are appended into them along with their respective IDs. The faces are then trained with their respective IDs.

The input image is read by the camera of the phone. After the image is read it is converted into gray scale. The faces in the image  are  detected  using  the  Haar  Cascade  frontal  face module. Using the LBPH algorithm, the faces in the image are predicted.  After the images are predicted, the recognized faces are shown in a green box along with their names. 


## CODE OVERVIEW
```
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# from PIL import ImageGrab

```

```
path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
``` 
 
```
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

```

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/images/76d.png" height="400px" align="left"/>
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
**`Open Source First`**
<p>We build projects to provide learning environments, deployment and operational best practices, performance benchmarks, create documentation, share networking opportunities, and more. Our shared commitment to the open source spirit pushes Face-x projects forward.</p>

