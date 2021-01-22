# Overview

## *CNN -> Convolutional Neural Network*

It is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. 

**The role of the CNN is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction.**

CNN models to train and test, each input image will pass it through a series of convolution layers with filters (Kernals), Pooling, fully connected layers (FC) and apply Softmax function to classify an object with probabilistic values between 0 and 1.

# Requirments

* Python-3
* Keras
* Numpy
* OpenCv

# Images

<p align="center"><img src="Images/Screenshot from 2020-12-11 21-34-18.png" height="250px">
<img src="Images/Screenshot from 2020-12-11 17-59-00.png" height="250px">
</p>

# Quick-Start

- Fork the repository
>click on the uppermost button <img src="https://github.com/Vinamrata1086/Face-X/blob/master/Recognition-Algorithms/Facial%20Recognition%20using%20LBPH/images/fork.png" width=50>

- Clone the repository using-
```
git clone https://github.com/akshitagupta15june/Face-X.git
```
- Create virtual environment-
```
- `python -m venv env`
- `source env/bin/activate`  (Linux)
- `env\Scripts\activate`  (Windows)
```
- Install dependencies-

- Headover to Project Directory- 
```
cd Recognition using Cnn

```
- Create dataset using -
```
- Run dataset.py on respective idle(VS Code, PyCharm, Jupiter Notebook, Colab)
```
Note: Do split the dataset into Train and Test folders.

- Train the model -
```
- Run train_model.py
```
Note: Make sure all dependencies are installed properly.

- Final-output -
```
- Run output.py
```

Note: Make sure you have haarcascade_frontalface_default.xml file 
