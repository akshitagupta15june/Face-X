### Introduction:
- Face mask detection had seen significant progress in the domains of Image processing and Computer vision, since the rise of the Covid-19 pandemic. Many face detection models have been created using several algorithms and techniques. The approach in this project uses deep learning, TensorFlow, Keras, and OpenCV to detect face masks.
- Convolutional Neural Network, Data augmentation are the key to this project.
### Example:
![face mask sample](https://raw.githubusercontent.com/sudipg4112001/Face-X/master/Face-Mask-Detection/Sample-images/Sample_image_1.jpg)
![face mask sample](https://raw.githubusercontent.com/sudipg4112001/Face-X/master/Face-Mask-Detection/Sample-images/Sample_image_2.jpg)

### Methodology used:

In order to train a custom face mask detector, I need to split our project into two distinct phases: 
- **Training Phase** (Training the model): <br>
  In this I will focus on loading our face mask detection datasets from disk, training a 
model using CNN on the datasets, and then serializing the face mask detector to disk.

- **Deployment Phase** (Deployment of the model): <br>
Once the face mask detector is trained, then move on to loading the mask detector, 
performing the face mask detection and then identifying each face as with mask or 
without mask.

![image](https://user-images.githubusercontent.com/51924622/184957736-17a83cf9-ff24-44bb-b444-fef5df1e4a8a.png)
![face mask sample](https://raw.githubusercontent.com/sudipg4112001/Face-X/master/Face-Mask-Detection/Sample-images/Method.jpg)

## Graphical Representation

![image](https://user-images.githubusercontent.com/51924622/184959136-4a8aac1a-19c0-419e-b59c-469b6e126346.png)

![image](https://user-images.githubusercontent.com/51924622/184959604-05f77e68-d387-49be-9c7d-7ae7ec237fbf.png)

## Classification Report

A Classification report is used to measure the quality of predictions from a classification 
algorithm. How many predictions are True and how many are False. More specifically, True 
Positives, False Positives, True negatives and False Negatives are used to predict the metrics 
of a classification report as shown below <br>

![image](https://user-images.githubusercontent.com/51924622/184960450-ef1371d0-7de2-428b-abb9-1591c006af84.png)


## Result

The accuracy of the model is **96%** after trained and tested on the images.

* __INPUT Img:__
![41698](https://user-images.githubusercontent.com/51924622/96024304-2f255080-0e71-11eb-99b8-ebeb3a8cc03c.jpg)

* __OUTPUT Img:__
![result1](https://user-images.githubusercontent.com/51924622/96024381-47956b00-0e71-11eb-9994-5816814a0200.png)

* __Another OUTPUT Img:__

![result](https://user-images.githubusercontent.com/51924622/96024389-495f2e80-0e71-11eb-8419-e9a21f71daba.png)

### This is the step by step methodology of how this project is created..!!
