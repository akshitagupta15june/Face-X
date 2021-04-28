<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/Untitled%20(3).png"  width="100%" /></a><br /><br /></p>




# Introduction
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/intersex%20teens-01.png" height="60%"  width="50%"  align="right"/>
In these modern days, gender recognition from facial image has been a crucial topic. To solve 
such delicate problem several handy approaches are being studied in Computer Vision. However, 
most of these approaches hardly achieve high accuracy and precision. Lighting, illumination, 
proper face area detection, noise, ethnicity and various facial expressions hinder the correctness 
of the research. Therefore, we propose a simple gender recognition system from facial image 
where we first detect faces from a scene using Haar Feature Based Cascade Classifier then introducing it to the model architecture. The face detection goal is achieved by OpenCV.



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
