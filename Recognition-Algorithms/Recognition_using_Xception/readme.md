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

### Contents

1. **requirements.txt**: A list of libraries that have to be included for the model to run. 

2. **output.py**: A file to check for the correctness of the model implementation. It recognises faces and outputs matching (if the face matched the name on which it was trained) or not accordingly as per trained data.

3. **create_dataset.py**: A script to detect faces by taking users input through the web camera and categorising them into train or val directory.

4. **main.py**: An example script to train an Xception model on the faces dataset.

5. **xception.py**: The model implementation file. Also required for preprocessing for inception models.

6. **__init__.py**: Initialise script by importing everything from xception.py

7. **Datasets**: A folder containing the faces dataset.

### How does Recognition using Xception work?

Xception is a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.
This architecture slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a **more efficient use of model parameters.**


Before moving on to learning about Depthwise Seperable Convolution , let us refresh our minds and get a quick idea of what convolution is through this GIF.

![](Convolution_of_box_signal_with_itself.gif)

In this example, the red-colored "pulse", is an even function so convolution is equivalent to correlation. A snapshot of this "movie" shows functions g(t-tau) and f(tau)(in blue) for some value of parameter t, which is arbitrarily defined as the distance from the tau = 0 axis to the center of the red pulse. The amount of yellow is the area of the product f(tau) and g(t-tau), computed by the convolution/correlation integral. The movie is created by continuously changing t and recomputing the integral. The result (shown in black) is a function of t, but is plotted on the same axis as tau, for convenience and comparison.

There are mainly 2 types of seperable convolutions-

1. Spatial Separable Convolution

2. Depthwise Seperable Convolution

To make it short and precise we will only discuss about depthwise seperable convolutions.

### What is Depthwise Seperable Convolutions (Main Principle behind Xception)

The depthwise separable convolution is so named because it deals not just with the spatial dimensions, but with the depth dimension as well. An input image may only have 3 channels: RGB. But after a few convolutions, an image may have multiple channels. You can image each channel as a particular interpretation of that image; in for example, the “red” channel interprets the “redness” of each pixel, the “blue” channel interprets the “blueness” of each pixel, and the “green” channel interprets the “greenness” of each pixel. An image with 64 channels has 64 different interpretations of that image. So unlike spatial separable convolutions, depthwise separable convolutions work with kernels that cannot be “factored” into two smaller kernels.

[How Kernels are iterated](https://www.youtube.com/watch?v=D_VJoaSew7Q)

### The main difference between normal convolution and depthwise is :- 
In the normal convolution, we are transforming the image 256 times. And every transformation uses up 5x5x3x8x8=4800 (where 5x5x3x256 represents the height, width, number of input channels, and number of output channels of the kernel) multiplications. In the separable convolution, we only really transform the image once — in the depthwise convolution. Then, we take the transformed image and simply elongate it to 256 channels. Without having to transform the image over and over again, we can save up on computational power.


### The modified depthwise separable convolution is the pointwise convolution followed by a depthwise convolution as one can see from the image below. This modification is motivated by the inception module in Inception-v3 that 1×1 convolution is done first before any n×n spatial convolutions.

![Modified Depthwise Separable Convolution in Xception](https://miro.medium.com/max/875/1*J8dborzVBRBupJfvR7YhuA.png)


In Xception, the modified depthwise separable convolution, there is NO intermediate ReLU non-linearity. This meant that when modified depthwise separable convolution with different activation units were tested. Xception without any intermediate activation had the highest accuracy compared with the ones using either ELU or ReLU.

### Overall Architecture of Xception
![Overall Architecture of Xception](https://miro.medium.com/max/875/1*hOcAEj9QzqgBXcwUzmEvSg.png)

Now let us see the program in action
## Screenshots
1) Firstly the input image is preprocessed.
2) Datasets are loaded and the program iterates through them.
3) The model with the best match is labeled as best model and trained for further outputs.
4) The model is then saved.


![Screenshot-1](https://user-images.githubusercontent.com/53366877/110513516-533d4300-812c-11eb-9cde-7566de26682f.png)

In the above screenshot, program was initially trained on the dataset of the the user (Harshit) and therefore it proceeds with the above mentioned steps and matches with the model that returned as the best match. The best weighted model for this case matched user with a string "Harshit" and hence "Face Found: Harshit" was printed on the screen.

![Screenshot-2](https://user-images.githubusercontent.com/53366877/110513613-6ea84e00-812c-11eb-86ec-d3fcecf921be.png)

In the above screenshot, there is no face present and therefore the dataset cannot match it with any of the trained models. This in turn returns a string which prints out "No face found" on the monitor of the user.
