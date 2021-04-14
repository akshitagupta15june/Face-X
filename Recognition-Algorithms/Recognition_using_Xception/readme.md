## Overview
**Xception is a convolutional neural network architecture that relies solely on depthwise separable convolution layers.**

Implementation of the Xception Model by François Chollet, based on the paper:

[Xception: Deep Learning with Depthwise Separable Convolutions.](https://arxiv.org/abs/1610.02357)

This model uses Xception model for the recognition of the User face using OpenCV and PyTorch.

The program is trained for 5 epochs, you can increase the number of epochs and the number of layers accordingly.


### Dependencies:
* PyTorch version **1.2.0** (get from https://pytorch.org/)

* Download haarcascades file from here=> https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

## Quick Start

- Fork and Clone the repository using-
```
git clone https://github.com/akshitagupta15june/Face-X.git
```
- Create virtual environment-
```
- `python -m venv env`
- `source env/bin/activate` (Linux)
- `env\Scripts\activate` (Windows)
```
- Install dependencies-
```
 pip install -r requirements.txt
```

- Headover to Project Directory- 
```
cd "Recognition using NasNet"
```
- Create dataset using -
```
 python create_dataset.py on respective idle(VS Code, PyCharm, Jupiter Notebook, Colab)
```
Note: Dataset is automatically split into train and val folders.

- Train the model -
```
 python main.py
```
Note: Make sure all dependencies are installed properly.

- Final-output -
```
 python output.py
```
Note: Make sure you have haarcascade_frontalface_default.xml file 

### How does Recognition using Xception work?

Xception is a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.
This architecture slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a **more efficient use of model parameters.**


### Contents

1. **requirements.txt**: A list of libraries that have to be included for the model to run. 

2. **output.py**: A file to check for the correctness of the model implementation. It recognises faces and outputs matching (if the face matched the name on which it was trained) or not accordingly as per trained data.

3. **create_dataset.py**: A script to detect faces by taking users input through the web camera and categorising them into train or val directory.

4. **main.py**: An example script to train an Xception model on the faces dataset.

5. **xception.py**: The model implementation file. Also required for preprocessing for inception models.

6. **__init__.py**: Initialise script by importing everything from xception.py

7. **Datasets**: A folder containing the faces dataset.

## Screenshots

![Screenshot from 2020-12-11 21-34-18](https://user-images.githubusercontent.com/53366877/110513516-533d4300-812c-11eb-9cde-7566de26682f.png)

![Screenshot from 2020-12-11 17-59-00](https://user-images.githubusercontent.com/53366877/110513613-6ea84e00-812c-11eb-86ec-d3fcecf921be.png)
