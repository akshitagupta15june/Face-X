# Overview

Dlib's facial recognition functionality is used by Adam Geitgey to build [face_recognition](https://github.com/ageitgey/face_recognition) library. It has an accuracy of **99.38%**. With the help of this library, we can convert the images in a dataset to the face encodings. At the time of face recognization, we can compare the unknown image's face encoding to the encodings present in the dataset to find the person.

# Requirments

- ```pip install dlib```
- ```pip install numpy```
- ```pip install git+https://github.com/ageitgey/face_recognition_models```

# Execution

- Clone the repository using-
```
git clone https://github.com/akshitagupta15june/Face-X.git
```
- Change Directory
```
cd Recognition-Algorithms/Recognition Using Dlib
```
- Add all the known images into the folder `images` and follow the steps given in `face.py`.

- Take one unknown image and give it to the program, it will fetch the name of the person present in the image if it's already in the dataset.

> **_NOTE:_**  This program can work for images that consist of only one person facing in front.