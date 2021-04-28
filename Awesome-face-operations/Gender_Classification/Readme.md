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



We are going to use Haarcascade and OpenCV to detect faces in a live webcam input stream. Then, we will retrain an inception v3 Artificial Neural Network to classify Male and Female faces. As training data, we are going to scrape some images from Bing Images search. Afterwards, we will use this slow inception v3 model to classify a big dataset of about 15'000 face images automatically, which we will then use to train a much faster Neural Network which will enhance the performance of the live classifier significantly.


# Cascade Face Detection









# Code OverView :
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os
import cv2
from IPython.display import Image
from sklearn.model_selection import train_test_split


data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])


path='../input/utkface-new/UTKFace/'
files=os.listdir(path)

image=cv2.imread(path+files[0])
image=np.expand_dims(image, 0)

X_data =[]
for file in files:
    face = cv2.imread(path +file,cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face =cv2.resize(face,(32,32))
    X_data.append(face)
X_data=np.array(X_data)
X_data.shape


aug_X=[]
for image in X_data:
    image=np.expand_dims(image, 0)
    for i in range(3):                   #the number of iteration here is adjustabe according to your dataset and the model
        aug_image=data_augmentation(image)
     aug_X.append(aug_image)


data=os.listdir("../input/utkface-new/UTKFace/")
y=[i.split("_")[1] for i in data ]
y=np.array(y,dtype=int)
y=np.repeat(y, 3)

X=np.squeeze(aug_X)

X_train,X_valid,Y_train,Y_valid=train_test_split(X,y,test_size=0.33)
print(X_train.shape,"\t", X_valid.shape,"\t", Y_train.shape,"\t",Y_valid.shape)

def model_gen():
    model=tf.keras.Sequential()
    model.add(layers.Conv2D(32,2,activation='relu',input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32,4,activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64,2,activation='relu'))
    model.add(layers.Conv2D(64,4,activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(84,2,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid',name='Gender_Out'))
    model.compile(optimizer='Adamax',loss=['binary_crossentropy'],
                  metrics=['accuracy'])
    tf.keras.utils.plot_model(model, 'model.png',show_shapes=True)

    return model

model=model_gen()
Image('model.png')

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss',restore_best_weights=True) ]
    
history=model.fit(X_train, Y_train, epochs=90,batch_size=240,
          validation_data=(X_valid,Y_valid),callbacks=callbacks, shuffle=True)
 
model.evaluate(X_valid,Y_valid)

for p_id in range(600 , 800 , 10):
    plt.imshow(X_valid[p_id])
    plt.show()
    print(Y_valid[p_id])
    #print(y_valid[0][p_id],y_valid[1][p_id])
    print(model.predict(np.expand_dims(X_valid[p_id],axis=0))[0][0])
    #np.expand_dims(X_valid[p_id],axis=0)

```


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
