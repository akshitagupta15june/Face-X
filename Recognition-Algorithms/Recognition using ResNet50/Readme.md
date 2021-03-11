## Overview
Face Recognition Using opencv, keras and tensorflow.

This model uses ResNet50 model for the recognition of the  User face.

Program is trained for 5 epochs, You can increase the number of epochs and the number of layers accordingly.

## ScreenShots

<img src="Screenshot from 2020-12-11 21-40-08.png" height="250px">
<img src="Screenshot from 2020-12-11 17-59-00.png" height="250px">

### Dependencies:
* pip install numpy
* pip install pandas
* pip install tensorflow
* pip install keras
* pip install opencv-python

Download haarcascades file from here=> https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml


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

- Headover to Project Directory- 
```
cd Recognition using ResNet50
```
- Create dataset using -
```
- Run Building_Dataset.py on respective idle(VS Code, PyCharm, Jupiter Notebook, Colab)
```
Note: Do split the dataset into Train and Test folders.

- Train the model -
```
- Run Training the model.py
```
Note: Make sure all dependencies are installed properly.

- Final-output -
```
- Run final_output.py
```
Note: Make sure you have haarcascade_frontalface_default.xml file 
### Details about Resnet 50
ResNet is a short name for Residual Network. As the name of the network indicates, the new terminology that this network introduces is residual learning. In a deep convolutional neural network, several layers are stacked and are trained to the task at hand. The network learns several low/mid/high level features at the end of its layers. In residual learning, instead of trying to learn some features, we try to learn some residual. Residual can be simply understood as subtraction of feature learned from input of that layer. ResNet does this using shortcut connections (directly connecting input of nth layer to some (n+x)th layer. It has proved that training this form of networks is easier than training simple deep convolutional neural networks and also the problem of degrading accuracy is resolved.
<br>
This is the fundamental concept of ResNet.
<br>
ResNet50 is a 50 layer Residual Network. There are other variants like ResNet101 and ResNet152 also.

