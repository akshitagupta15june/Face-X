<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/Untitled%20(3).png"  width="100%" /></a><br /><br /></p>




# Introduction
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/intersex%20teens-01.png" height="70%"  width="60%"  align="right"/>
In these modern days, gender recognition from facial image has been a crucial topic. To solve 
such delicate problem several handy approaches are being studied in Computer Vision. However, 
most of these approaches hardly achieve high accuracy and precision. Lighting, illumination, 
proper face area detection, noise, ethnicity and various facial expressions hinder the correctness 
of the research. Therefore, we propose a simple gender recognition system from facial image 
where we first detect faces from a scene using Haar Feature Based Cascade Classifier then introducing it to the model architecture. The face detection goal is achieved by OpenCV.


Automatic gender recognition has now pertinent to an extension of its usage in various software and hardware, particularly because of the growth of online social networking websites and social media. However the performance of already exist system with the physical world face pictures, images are somewhat not excellent, particularly in comparison with the result of task related to face recognition. Within this paper, we have explored that by doing learn and classification method and with the utilization of Deep Convolutional Neural Networks (D-CNN) technique, a satisfied growth in performance can be achieved on such gender classification tasks that is a reason why we decided to propose an efficient convolutional network VGGnet architecture which can be used in extreme case when the amount of training data used to learn D-CNN based on VGGNet architecture is limited. We examine our related work on the current unfiltered image of the face for gender recognition and display it to dramatics outplay current advance updated methods

# Problem Identification & Definition
Gender classification has gained importance in recent times due to increasing influence and rise of social media
platforms.
However, accuracy of previous algorithms on the images is still not sufficient enough Which can match the performance
made in the field of face recognition.
Still this problem is a tricky problem which needs to be resolved . The main difficulty is that the
Nature as well as the abundancy of data which is required to train the type of systems. While general classification projects
have access to millions of images which is very helpful in training but for  gender classification specifically we have
lesser number of images maybe in the range of thousands. The main reason behind this is that in order to have labels we
should have personal data of the subjects in the images.
Hence we require new algorithms and tools to cope up with this problem. These above reasons are responsible for
choosing this new approach

# Gender Detection


#### Table of contents

- Introduction
- Dataset
- Implementation
- Outputs
- Running procedure
- Dependencies


### Introduction

In these modern days, gender recognition from facial image has been a crucial topic. To solve 
such delicate problem several handy approaches are being studied in Computer Vision. However, 
most of these approaches hardly achieve high accuracy and precision. Lighting, illumination, 
proper face area detection, noise, ethnicity and various facial expressions hinder the correctness 
of the research. Therefore, we propose a simple gender recognition system from facial image 
where we first detect faces from a scene using Haar Feature Based Cascade Classifier then introducing it to the model architecture. The face detection goal is achieved by OpenCV.


### Dataset
![logoWall2](https://user-images.githubusercontent.com/55057549/112679952-7169a980-8e75-11eb-8e64-e83997864119.jpg)
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization.

link to download dataset: https://www.kaggle.com/jangedoo/utkface-new



### Implementation

- #####   Model Architecture
 ![model](https://user-images.githubusercontent.com/63206325/113521830-4af5ed80-959c-11eb-9822-ecc8475f27cc.png)

- #####   Data Augmentation
  Data augmentation can be used to address both the requirements, the diversity of the training data, and the amount of data. Besides these two, augmented data can also be used to address the class imbalance problem in classification tasks.In order to increase the model ability to detect the gender from different point of views , we decided to use the data augmentation concept.

![image](https://user-images.githubusercontent.com/63206325/113521861-809ad680-959c-11eb-9e64-3de5b544dba6.png)



### Outputs

![image](https://user-images.githubusercontent.com/63206325/113521952-4bdb4f00-959d-11eb-9af6-36e422919f74.png)
![image](https://user-images.githubusercontent.com/63206325/113521963-5990d480-959d-11eb-8649-457005a0031e.png)
![image](https://user-images.githubusercontent.com/63206325/113521969-631a3c80-959d-11eb-80db-53381a3a35af.png)
![image](https://user-images.githubusercontent.com/63206325/113521972-6c0b0e00-959d-11eb-8a8d-bccc183e879b.png)




### Dependencies
- tensorflow 2.4.1
- openCV
- Numpy
- OS
- Matplotlib


### Running Procedure
- Clone the Repository 
- Open your notebook
- check paths for the test data
- Enjoy the experience 
