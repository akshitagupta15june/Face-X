## Overview
Face Recognition Using OpenCV and PyTorch.

This model uses GhostNet model for the recognition of the  User face.

Program is trained for 5 epochs, You can increase the number of epochs and the number of layers accordingly.


### Dependencies:
* pip install numpy
* pip install Pillow
* pip install tqdm
* pytorch (get from https://pytorch.org/)
* pip install opencv-python

Download haarcascades file from here=> https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

## ScreenShots

<img src="Screenshot 2021-01-15 115306.png" height="250px">
<img src="Screenshot 2021-01-15 115354.png" height="250px">


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
cd "Recognition using GhostNet"
```
- Create dataset using -
```
- Run create_dataset.py on respective idle(VS Code, PyCharm, Jupiter Notebook, Colab)
```
Note: Dataset is automatically split into train and val folders.

- Train the model -
```
- Run train-model.py
```
Note: Make sure all dependencies are installed properly.

- Final-output -
```
- Run output.py
```
Note: Make sure you have haarcascade_frontalface_default.xml file 
