## Overview
Face Recognition Using OpenCV and PyTorch.

This model uses Xception model for the recognition of the  User face.

Program is trained for 5 epochs, You can increase the number of epochs and the number of layers accordingly.


### Dependencies:
* pytorch version **1.2.0** (get from https://pytorch.org/)


Download haarcascades file from here=> https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

## ScreenShots

![Screenshot from 2020-12-11 21-34-18](https://user-images.githubusercontent.com/53366877/110513516-533d4300-812c-11eb-9cde-7566de26682f.png)

![Screenshot from 2020-12-11 17-59-00](https://user-images.githubusercontent.com/53366877/110513613-6ea84e00-812c-11eb-86ec-d3fcecf921be.png)



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
