<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/Readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/Attendance%20system/images/932_generated.png"  width="100%" /></a><br /><br /></p>

# Facial Recognition Attendance Management System 

# INTRODUCTION
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
1. It is one of the simplest algorithms for face recognition.
2. The local features of the images can be characterized by this algorithm.
3. Using this algorithm, considerable results can be obtained. 
4. Open CV library is used to implement LBPH algorithm
