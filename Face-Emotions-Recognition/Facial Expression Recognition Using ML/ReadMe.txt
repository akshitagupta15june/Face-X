
### Overview

* Facial expression recognition software is a technology which uses biometric markers to detect emotions in human faces. 
* More precisely, this technology is a sentiment analysis tool and is able to automatically detect the six basic or universal expressions: happiness, sadness, anger, neutral,  surprise, fear, and disgust.
* Facial expressions and other gestures convey nonverbal communication cues that play an important role in interpersonal relations.
* Therefore, facial expression recognition, because it extracts and analyzes information from an image or video feed, it is able to deliver unfiltered, unbiased emotional responses as data.

## Built With

* [Python](https://www.python.org/)
* [Jupyter Notebook](https://jupyter.org/)
* [Google Colab](https://colab.research.google.com/)

### Problem Statement

* Given a data set consisting of facial images and their sketches, retrieve all images (real and /or sketch) which are similar to the given test image, along with various attributes of the image such as gender, expression and so on. 

### Data Source

* The dataset was collected by us, consisting of 60 university students. 
* Total images = 60 * 7(happiness, sadness, anger, neutral,  surprise, fear, and disgust).

### Plan

* Face Detection - Locating faces in the scene, in an image or video footage. 
* Facial Landmark Detection - Extracting information about facial features from the detected faces. 
* Facial Expression And Emotion Classification - Classifying the obtained information into expression interpretative categories such as smile or frown or emotion categories such as happy, anger, disgust etc.  

## Approach

### Data Cleaning

* After importing the images, the images were resized to 420 × 240 because some of the images in the dataset did not have 1280 × 960 as their size, despite the submission format.

### Data Preprocessing 

* The images were then converted into grayscale to remove the third dimension and to make the implementation easier.
* Then the images were then flattened (except for CNN) and for Neural Network we have applied PCA to reduce image’s dimensions.
* Histogram of oriented gradients was used to extract faces from entire images. 
* Then the dataset was divided into two parts 90% of the dataset was used for training and rest 10% was used for testing.

### Data Augmentation

* We have used data augmentation to increase size of our dataset.

### Learning Algorithms 

* We have taken two types of approaches:
  * Non-neural network approach 
    * K Nearest Neighbours (with k = 5, minkowski distance with p = 2)
    * Support Vector Machine (linear kernel)
    * Naive Bayes (Gaussian with variance 10^-9)
    * Decision Tree
    * Random Forest (n = 10)
  * Neural network approach
    * Back propagation Neural Network (with 15 features and 2 layers)
    * Convolutional Neural Network (3 convolutional layers and 2 fully connected layers with pooling layers)

## Results : 

![1](.images/1.png)

![2](.images/2.png)


## Download Model : 
* https://drive.google.com/drive/folders/1Iyc_MuTU9gjC8E34pDEmhmgWP8hMX8Fm?usp=sharing


## References

* Sharif M., Mohsin S., Hanan R., Javed M. and Raza M., ”Using nose Heuristics for Efficient face Recognition”, Sindh Univ. Res. Jour. (Sci. Ser.) Vol.43 (1-A), 63-68,(2011)
* Maryam Murtaza, Muhammad Sharif, Mudassar Raza, Jamal Hussain Shah, “Analysis of Face Recognition under Varying Facial Expression: A Survey”, The International Arab Journal of Information Technology (IAJIT) Volume 10, No.4 , July 2013
* https://medium.com/neurohive-computer-vision/state-of-the-art-facial-expression-recognition-model-introducing-of-covariances-9718c3cca996/
* https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/ 
