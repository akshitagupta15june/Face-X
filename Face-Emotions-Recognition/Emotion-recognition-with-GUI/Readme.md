# Emotion Detection GUI
A deep learning project that aims at detecting using Human emotions.

# Dataset Used
The data set i used was FER2013 competition data set  for emotion detection. The dataset has 35685 images. All of these images are of size 48x48 pixel and are in grayscale. It contains images of 7 categories of emotions. </br>

FER 2013 emotions data - https://www.kaggle.com/ananthu017/emotion-detection-fer

# Approach
* I started off by using the FER2013 data set and carried out some data preprocessing and data augmentation on it.
* I experimented with many models - VGG16, ResNet, Xception and so on. None of them seemed to work very well.
* I then went on to build my own custom model. The model was built using Tensorflow and Keras and had the following layers - 
    * Conv2D 
    * MaxPooling2D
    * Batch Normalization
    * Dense
    * Flatten
    * Dropout  
* After lot of experiments with custom architectures, i was able to attain a model that gave us decent results.

# Results
The model was trained for 60 epochs and had a batch size of 64. We were able to achieve an accuracy of 71% on training set and a validation accuracy of 65%.

# Image gallery

* Main GUI window without result - 
<p align="center">
<img src="" alt="drawing" width="300"/>
</p>

* Some Model results - 
<p align="center">
<img src="" alt="drawing" width="300"/>
</p>
<p align="center">
<img src="" alt="drawing" width="300"/>
</p>
<p align="center">
<img src="" alt="drawing" width="300"/>
</p>

* Main GUI window with result -
<p align="center">
<img src="" alt="drawing" width="300"/>
</p>

# Model

The model was a bit big so i uploaded it in a drive folder and you can get it via this link - 
https://drive.google.com/drive/folders/10pmRx3ZVEt1r2zEWsAtiP8BJGwTgH0OF?usp=sharing