# Video Face Recognition

What if you could know whether a person is present in a video or not, without seeing the video. That is exactly what's done here.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/media3.gif" width="400">
</p>

The only libraries used here are OpenCV and Sci-Kit Learn. Streamlit is used to make a GUI.

This task is acheived by the following steps:
1. Detect Faces
2. Compute 128-d face embeddings to quantify a face
3. Train a Support Vector Machine (SVM) on top of the embeddings
4. Recognize faces in the video stream

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/media1.jpeg" width="400" height="300">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/media2.jpeg" width="400" height="300">
</p>

## Detect Faces

The OpenCv hidden face detector is used to detect faces. There are 2 files used for this - 
  1. deploy.prototxt.txt - The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
  2. res10_300x300_ssd_iter_140000.caffemodel - The .caffemodel file which contains the weights for the actual layers

OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network.


## Computing 128-d face embeddings to quantify a face

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/media5.jpg" width="400">
</p>

The model responsible for actually quantifying each face in an image is from the OpenFace project, a Python and Torch implementation of face recognition with deep learning. It is called FaceNet.

The FaceNet deep learning model computes a 128-d embedding that quantifies the face itself.
To train a face recognition model with deep learning, each input batch of data includes three images:

  - The anchor - is the current face and has identity A  
  - The positive image - this image also contains a face of person A  
  - The negative image - does not have the same identity, and could belong to person B, C, or even Y  
  
The point is that the anchor and positive image both belong to the same person/face while the negative image does not contain the same face.
The neural network computes the 128-d embeddings for each face and then tweaks the weights of the network such that:
  - The 128-d embeddings of the anchor and positive image lie closer together
  - While at the same time, pushing the embeddings for the negative image father away


## Training a Support Vector Machine (SVM) on top of the embeddings

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/media6.jpg" width="400">
</p>


### The advantages of support vector machines are:

  - Effective in high dimensional spaces.  
  - Still effective in cases where number of dimensions is greater than the number of samples.  
  - Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.  
  - Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.  
  
### The disadvantages of support vector machines include:

  - If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.  
  - SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.  

In this application we use the "Linear" kernel of SVC(Support Vector Classifier)


## Recognize faces in the video stream

Finally, an input video and input photos of a person are taken along with their name. Once confirmed, the video is displayed with boxes around the input face's face in the video.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/meida4.gif" width="400">
</p>

## Demo

Here is a demo on how the scripts work

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/demo.gif" width="800">
</p>


## Dependencies

 - OpenCV - This is used to read the images and videos. It is also used to load the FaceNet model to quantify each face i.e, to calculate the 128-d face embeddings
 - Sci-Kit Learn - This is used to train the SVM Classifier on the face embeddings.
 - Imutils - This is used to resize the images while maintaining the aspect ratio.
 - Os - This is used for file writing and management
 - Streamlit - This is used to make the GUI for face recognition
 - Numpy - This is used to read the files given as input in streamlit
 - Pickle - This is used to store unknown embeddings and unknown names (Discussed below)


## Setup

- Fork the repository - Creates a copy of this project in your github.

- Clone the repository to your local machine using 
```
git clone https://github.com/akshitagupta15june/Face-X.git
```
- Use a virtual environment to keep the all dependencies in a separate enviroment for example - conda, virtualenv, pipenv, etc.

- Navigate to the Differentiate between Human and Cartoon Faces inside Cartoonify Image Folder using
```
cd Video-Face-Recognition
```
  
- Install the dependencies either by using the below pip commands or by using the requirements.txt file given.

- By using pip commands
```
pip install numpy
```
```
pip install opencv-python
```
```
pip install imutils
```
```
pip install -U scikit-learn
```
```
pip install streamlit
```

- By using requirements.txt
```
pip install -r requirements.txt
```

- Run the run.py script using
```
streamlit run run.py
```

## How does it work?

- Upload a video
- Enter the number of faces to recognize
- For each face, type the name and press enter. After that upload photos of each person (Minimum 6) and click "Confirm".
- Wait for 2 seconds and voila! The video is displayed with boxes around the faces recognized. Along the boxes, the name of the face recognized and its confidence score is displayed. Faces that aren't recognized are labeled as "Unknown".

## How are Unknown Faces recognized

There are 2 extra files in this directory - unknownEmbeddings.pkl and unknownNames.pkl. 
unknownEmbeddings.pkl - Contains embeddings of random faces
unknownNames.pkl - Contains the word "Unknown" for each random face chosen

These 2 files (or lists) are merged with the embeddings and names calculated from the user's input and then are collectively trained by the SVM. So everytime, one uses this script, the unknown face data is automatically added.


## Want to Contribute?

- Follow the steps for Setup

- Make a new branch
```
git branch < YOUR_USERNAME >
```

- Switch to Development Branch
```
git checkout < YOURUSERNAME >
```

- Make a folder and add your code file and a readme file with screenshots.

- Add your files or changes to staging area
```
git add.
```

- Commit Message
```
git commit -m "Enter message"
```

- Push your code
```
git push
```

- Make Pull request with the Master branch of akshitagupta15june/Face-X repo.

- Wait for reviewers to review your PR
