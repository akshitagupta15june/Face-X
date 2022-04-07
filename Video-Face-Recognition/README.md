# Video Face Recognition

What if you could know whether a person is present in a video or not without seeing the video. That is exactly what's done here.

The only libraries used here are OpenCV and Sci-Kit Learn. Streamlit is used to make a GUI.

This task is acheived by the following steps:
1. Detect Faces
2. Compute 128-d face embeddings to quantify a face
3. Train a Support Vector Machine (SVM) on top of the embeddings
4. Recognize faces in the video stream

## Detect Faces

The OpenCv hidden face detector is used to detect faces. There are 2 files used for this - 
  1. deploy.prototxt.txt - The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
  2. res10_300x300_ssd_iter_140000.caffemodel - The .caffemodel file which contains the weights for the actual layers

OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network.


## Computing 128-d face embeddings to quantify a face

The model responsible for actually quantifying each face in an image is from the OpenFace project, a Python and Torch implementation of face recognition with deep learning. It is called FaceNet.

The FaceNet deep learning model computes a 128-d embedding that quantifies the face itself.
To train a face recognition model with deep learning, each input batch of data includes three images:

  · The anchor - is the current face and has identity A  
  · The positive image - this image also contains a face of person A  
  · The negative image - does not have the same identity, and could belong to person B, C, or even Y  
  
