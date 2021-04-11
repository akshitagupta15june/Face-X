## Facial-Features-and-Face-Detection-CNN
**This project is for building CNN architecture for facial features and face detection by plotting points on the facial features and box around the face**

## Data Source:
In the following link, there is a dataset from kaggle website and all the necessary information that will be needed to understand the nature of the data:
[**Dataset Source** ](https://www.kaggle.com/jessicali9530/celeba-dataset)
  
But we are going to use only **the data of the first 35000 images** beacuse the data is **very big (1 GB)** for the memory.

## Performace of the model:
The accuracy of the model = 90%

## Content of this repository:
#### 1- data
In this folder, there are:
* 35000 images
* 60 test images for just showing the performace on the final model
* CSV file contains the keypoints of facial features as (x,y) coordinates and the image_id
#### 2- jupyter notebook of the model.
In this notebook, i build the CNN architecture of the model.  
#### 3- model.h5
This is the weights of the final model
#### 4- jupyter notebook of the testing.
In this notebook, we test the model on 60 images to show the performace of the model
