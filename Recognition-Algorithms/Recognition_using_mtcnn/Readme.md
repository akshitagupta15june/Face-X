## Quick Start
- Clone this repository.
`git clone https://github.com/akshitagupta15june/Face-X.git`
- Change Directory.
`cd Recognition_using_mtcnn`
- Run File.
`python mtcnn.py`
## Dependencies
- `pip install tensorflow`
- `pip install mtcnn`
- `pip install matplotlib`
# Screenshots
![Capture](https://github.com/akshitagupta15june/Face-X/blob/master/Recognition-Algorithms/Recognition_using_mtcnn/img/ouput.png)
![Capture1](https://github.com/akshitagupta15june/Face-X/blob/master/Recognition-Algorithms/Recognition_using_mtcnn/img/ouput1.png)


# Face Recognition with MTCNN

Deep learning advancements in recent years have enabled widespread use of face recognition technology. This project tries to explain deep learning models used for face recognition and introduces a simple framework for creating and using a custom face recognition system.

Face recognition can be divided into multiple steps. The image below shows an example of a face recognition pipeline.

![alt text](https://arsfutura-production.s3.us-east-1.amazonaws.com/magazine/2019/10/face_recognition/face-recognition-pipeline.png "Image of Face Recognition is done")

# What is MTCNN:

Multi-Task Cascaded Convolutional Neural Networks(MTCNN) is a neural network which detects faces and facial landmarks on images.

## Features of MTCNN:

1. Consists of 3 neural networks connected in a cascade.
2. One of the most popular and most accurate face detection tools today.

# How to Implement MTCNN:

1. After cloning the repo from github (link here),  open up and run mtcnn.py which produces the image:
    ![alt text](https://github.com/akshitagupta15june/Face-X/raw/master/Recognition-Algorithms/Recognition_using_mtcnn/img/ouput1.png "Image of Face Recognition is done")

2. As seen in the image above, the neural network detects individual faces, locates facial landmarks (i.e. two eyes, nose, and endpoints of the mouth), and draws a bounding box around the face. The code from mtcnn.py supports this.

## How does the algorithm works:

1. First,  MTCNN. Checking ./mtcnn/mtcnn.py showed the MTCNN class, which performed the facial detection.
2. Then, a detector of the MTCNN class was created, and the image read in with cv2.imread. The detect_faces function within the MTCNN class is called, to “detect faces” within the image we passed in and output the faces in “result”.
3.Now we draw the rectangle of the bounding box by passing in the coordinates, the color (RGB), and the thickness of the box outline. Here, bounding_box[0] and bounding_box[1] represent the x and y coordinates of the top left corner, and bounding_box[2] and bounding_box[3] represent the width and the height of the box, respectively.
4. Similarly, we can draw the points of the facial landmarks by passing in their coordinates, the radius of the circle, and the thickness of the line.


# Concluding statements:

Running this new file, one can see that the MTCNN network can indeed run in real time. boxing his/her face and marking out features. 

