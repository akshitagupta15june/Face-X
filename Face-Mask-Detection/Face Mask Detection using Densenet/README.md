# Face-Mask-Detection
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

COVID-19 has made wearing face masks a part of everyone's daily lives. Making sure the people wear masks inside stores and public spaces has become a priority. As well, being able to determine if someone is wearing a mask is important for contact tracing and the transmission of COVID. This was the motivation to create a face mask detection model that can detect face masks in real-time. For this project, I coded in Python in a Jupyter Notebook and used TensorFlow, Keras, NumPy, and OpenCV.

## Dataset

The face mask dataset used was compiled by [Chandrika Deb](https://github.com/chandrikadeb7). 

The dataset has **3835 images** split between two classes: **1916 images** of faces with masks and **1919 images** of faces without masks.

I split the data into 75% training set, 10% validation set, and 15% testing set.

The dataset can be downloaded [here](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG).

## Results

The mask detection classifier reached **97% accuracy**.

![](mask-detection-demo.gif)

## Installation

