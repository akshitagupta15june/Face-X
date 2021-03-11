Face and Eye Detection using Haar Cascade:

Face and Eye detection is done using Haar Cascade. Haar Cascade is a
Machine Learning based approach where a cascade function is trained
with set of input data.It contains pre-trained classifiers for face,
eyes,smile etc. For face detection we have used a face classifier
and similarly for eye detection we have used an eye classifier.
Using Opencv we can detect faces and eyes in images as well as videos.

Step-1:Import all the necessary libraries like opencv.

Step-2:We have then added the path of classifiers(pre-trained) for 
face and eyes as face_cascade and eye_cascade respectively.

Step-3:Then we use a function named VideoCapture() to capture
our video using our camera.

Step-4:Video contains different frames,which are in BGR format.
These frames are coverted to Grey frames using cvtColor() function.

Step-5:Using the classifiers our face and eyes are detected.

Step-5:Using cv2.rectangle() function,a rectangle is created around our face and eyes
of any color we want.


