# Video Face Recognition

What if you could know whether a person is present in a video or not, without seeing the video. That is exactly what's done here.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Video-Face-Recognition/Media/media3.gif" width="400">
</p>

The only libraries used here are OpenCV and Sci-Kit Learn. Streamlit is used to make a GUI.

This task is acheived by the following steps:
1. Detect Faces
2. Compute 128-d face embeddings to quantify a face
3. Train a Support Vector Machine (SVM) on top of the embeddings
4. Recognize faces in the video stream

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Video-Face-Recognition/Media/media1.jpeg" width="400" height="300">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Video-Face-Recognition/Media/media2.jpeg" width="400" height="300">
</p>

## Detect Faces

The OpenCv hidden face detector is used to detect faces. There are 2 files used for this - 
  1. deploy.prototxt.txt - The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
  2. res10_300x300_ssd_iter_140000.caffemodel - The .caffemodel file which contains the weights for the actual layers

OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network.


## Computing 128-d face embeddings to quantify a face

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Video-Face-Recognition/Media/media5.jpg" width="400">
</p>

The model responsible for actually quantifying each face in an image is from the OpenFace project, a Python and Torch implementation of face recognition with deep learning. It is called FaceNet.

The FaceNet deep learning model computes a 128-d embedding that quantifies the face itself.
To train a face recognition model with deep learning, each input batch of data includes three images:

  · The anchor - is the current face and has identity A  
  · The positive image - this image also contains a face of person A  
  · The negative image - does not have the same identity, and could belong to person B, C, or even Y  
  
The point is that the anchor and positive image both belong to the same person/face while the negative image does not contain the same face.
The neural network computes the 128-d embeddings for each face and then tweaks the weights of the network such that:
  · The 128-d embeddings of the anchor and positive image lie closer together
  · While at the same time, pushing the embeddings for the negative image father away


## Training a Support Vector Machine (SVM) on top of the embeddings

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Video-Face-Recognition/Media/media6.jpg" width="400">
</p>


### The advantages of support vector machines are:

  · Effective in high dimensional spaces.  
  · Still effective in cases where number of dimensions is greater than the number of samples.  
  · Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.  
  · Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.  
  
### The disadvantages of support vector machines include:

  · If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.  
  · SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.  

In this application we use the "Linear" kernel of SVC(Support Vector Classifier)


## Recognize faces in the video stream

Finally, an input video and input photos of a person are taken along with their name. Once confirmed, the video is displayed with boxes around the input face's face in the video.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Video-Face-Recognition/Media/meida4.gif" width="400">
</p>

## Demo

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Video-Face-Recognition/Media/meida4.gif" width="400">
</p>
