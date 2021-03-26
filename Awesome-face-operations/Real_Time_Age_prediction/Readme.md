# Real-Time-Age Predictor
## Table of content

- Introduction
- Model Architecture
- Dataset
- Outputs
- Dependencies
- Running Procedure
# Introduction

![__opencv_age_detection_examples](https://user-images.githubusercontent.com/55057549/112669943-fa2e1880-8e68-11eb-996b-d5c7efddc6d8.jpg)

Age detection is the process of automatically discerning the age of a person solely from a photo of their face.
Typically, you’ll see age detection implemented as a two-stage process:

Stage #1: Detect faces in the input image/video stream
Stage #2: Extract the face Region of Interest (ROI), and apply the age detector algorithm to predict the age of the person

## Model Architecture

![__results___10_0](https://user-images.githubusercontent.com/55057549/112670801-0e264a00-8e6a-11eb-85a2-522bbedd8c65.png)

## Dataset
![logoWall2](https://user-images.githubusercontent.com/55057549/112679952-7169a980-8e75-11eb-8e64-e83997864119.jpg)

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc
__Link to dataset__ : https://susanqq.github.io/UTKFace/



## Outputs
Real Age = 35
![Capture](https://user-images.githubusercontent.com/55057549/112677174-0ff40b80-8e72-11eb-96a6-e846adfb80be.PNG)

Real Age = 85 
![Capture1](https://user-images.githubusercontent.com/55057549/112677632-aaece580-8e72-11eb-9e4b-5f18d2a29aeb.PNG)
## Dependencies
- tensorflow 2.4.1
- openCV
- Numpy
- OS
- Matplotlib


## Running Procedure
- Clone the Repository 
- Open your notebook
- check paths for the test data
- Enjoy the experience 



