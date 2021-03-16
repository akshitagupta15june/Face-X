## Transfer Learning for Image Classification in PyTorch

### Introduction:
- Face mask detection is a significant progress in the domains of Image processing and Computer vision, since the rise of the Covid-19 pandemic. Many face detection models have been created using several algorithms and techniques. The approach in this project uses deep learning, pytorch, numpy and matplotlib to detect face masks and calculate accuracy of this model.
- Transfer Learning, Data augmentation are the key to this project.

### Major features:

- How a CNN works
- Layer and classifier visualization
- Data preparation
- Modifying a Pretrained Model (ResNet34), using transfer learning

### Methodology used:
![face mask sample](./Sample-Images/model-image.png)

### Dependencies:
- opendatasets
- os
- torch
- torchvision
- numpy
- matplotlib

### Dataset Used:
We'll use the COVID Face Mask Detection Dataset dataset from [Kaggle](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset). This dataset contains about 1006 equally distributed images of 2 distinct types, namely `Mask` and `Non Mask`.

### Demo

Just head over to [face-mask-detection-pytorch.ipynb](Face-X/Face-Mask-Detection/face-mask-detection-pytorch/face-mask-detection-pytorch.ipynb), and run the python notebook on your local computer.


### Example:
![dataset sample](./Sample-Images/Sample-Image.png) 

### Results:
- Validation loss: 0.943358838558197
- Validation accuracy: 0.8799999952316284
