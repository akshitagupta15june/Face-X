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



# Deep convolutional neural networks : 
 In contrast to modern Deep Convolutional Neural Networks (D-CNN) like VGGNet, LeNet-5 network architecture
was comparatively moderate because of the limited number of computational resources such as time, memory, processing power,
and the various algorithmic challenges for training the huge networks. Yet much potential exists in Deep Convolutional Neural
Networks (D-CNN) architectures (a neural network with the huge number of neuron layers), recently they have become vogue,
because of an unexpected rise in both the computational power with the usage of Graphical Processing Units(GPU), and the
amount of dataset that is easily available on the Internet or prepares by a researcher in order to do practically. One of the biggest
examples used in the physical world with the use of Deep Convolution Neural Network(D-CNN) is for image classification,
Recognition on various facial database of million number of raw unfiltered face image like LFW
<img  src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/492682_1_En_10_Fig1_HTML.png"/>

Deep CNN(D-CNN) is using in this area also including Articulated pose estimation, Body configuration parsing, face parsing, Face
recognition, object detection, path detection, plant disease estimation through the image of plant leaves, age and expression recognition
through the face of human being, facial key-point detection, speech recognition.


# Composition of Deep Convolutional and Subsample layer : 

The number of layers involves the work of Deep Convolution Neural Network (D-CNN) is scaled down by the convolutional
layer and a subsample layer in a single layer. This concept was popularized by Simard, which was after known by Mamalet and
Garcia.In this task, we change back to back sub-sample layers and convolutional the single convolutional layer using two strides.
A pattern on an image can be extracted by the following expression:
where pi
(t-e) and pj
(t) are the input and output pattern map respectively, F() is function i.e known by activation
function which we have used in our work,mef
(t) is the convolutional kernel weight j
(t) represents bias denotes total
number of input feature mapping, sq
(t)q represents horizontal convolution step size, sp
(t)p represents vertical
convolution step size, and Rq
(t) and Rp
(t) are width and height of convolutional kernels, respectively. where M(te)and A(t-e) and height and width of input feature mapping

A(t) = (A(t-e) - Rp
(t))/sp
(t) + 1 (2)
M(t) = (M(t-e)- Rq
(t))/sq
(t) + 1 (3)
figure:5 depicts how CNN work with an image for Gender Recognition and give final unique output on the basis
of internal processing of pixels, patterns of image


# Gender classification. 
A detailed  of gender classification methods can be found in  and more recently
in . Here we quickly survey relevant methods.
One of the early methods for gender classification 
used a neural network trained on a small set of near-frontal
face images. In  the combined 3D structure of the
head (obtained using a laser scanner) and image intensities were used for classifying gender. SVM classifiers
were used by , applied directly to image intensities.
Rather than using SVM, used AdaBoost for the same
purpose, here again, applied to image intensities. Finally,
viewpoint-invariant age and gender classification was presented by.
More recently,used the Webers Local texture Descriptor  for gender recognition, demonstrating nearperfect performance on the FERET benchmark .
In , intensity, shape and texture features were used with
mutual information, again obtaining near-perfect results on
the FERET benchmark.
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/nltk-2.png" align="right" height="70%" width="70%"/>

The basic classification task has a number of interesting variants. For example, in multi-class classification, each instance may be assigned multiple labels; in open-class classification, the set of labels is not defined in advance; and in sequence classification, a list of inputs are jointly classified.
A classifier is called supervised if it is built based on training corpora containing the correct label for each input. The framework used by supervised classification



<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/cnn_illustration_a.png" align="right"/>
Illustration of our CNN architecture. The network contains three convolutional layers, each followed by a rectified linear
operation and pooling layer. The first two layers also follow normalization using local response normalization [28]. The first Convolutional
Layer contains 96 filters of 7×7 pixels, the second Convolutional Layer contains 256 filters of 5×5 pixels, The third and final Convolutional
Layer contains 384 filters of 3 × 3 pixels. Finally, two fully-connected layers are added, each containing 512 neurons. See Figure 3 for a
detailed schematic view and the text for more information.


# Cascade Face Detection

Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, “Rapid Object Detection using a Boosted Cascade of Simple Features” in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, haar features shown in below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.

# Configuring your development environment
To follow this guide, you need to have the OpenCV library installed on your system

```
OpenCV Face detection with Haar cascades
$ pip install opencv-contrib-python
```
#### Dependencies
- tensorflow 2.4.1
- openCV
- Numpy
- OS
- Matplotlib

#### Running Procedure
- Clone the Repository 
- Open your notebook
- check paths for the test data
- Enjoy the experience

# Training and testing :
he weights that we have used as a part of all layers are introduced with stochastic esteems from a mean Gaussian
with value zero and a standard deviation around 0.01.We don’t utilize pre-prepared profound Convolutional Neural
Network models for introducing the system; this system framework is trained, from a root, without utilizing any information outward of the pictures and the names accessible by the benchmark. This is again, should be contrasted
and profound Convolutional Neural Network(CNN) executions utilized for confronting acknowledgment with
gender, age, facial articulation, where a huge number of pictures are utilized for preparing.
We have used specific network architecture as well as the dataset. We are going to use a combination of
LogSoftMax + NLL-Loss in PyTorch to train the network.
Next, we load the pre-trained VGG-Face model and dataset. We have also initialized our network architecture, loss
module and certain other training parameters such as the number of epochs to train and the batch size. At first, we
fix the seed of the various random number generators that our code required to use


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
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization.

link to download dataset: https://www.kaggle.com/jangedoo/utkface-new

###   Data Augmentation
  Data augmentation can be used to address both the requirements, the diversity of the training data, and the amount of data. Besides these two, augmented data can also be used to address the class imbalance problem in classification tasks.In order to increase the model ability to detect the gender from different point of views , we decided to use the data augmentation concept.

### Network architecture
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/Gender-classification-network-architecture.png" align="right"/>
Images are scaled again to 256 x 256 size image and a then perform cropping operation on the image of size 227
x 227 which is passed into the network. The three consecutive convolutional layers are then described as
The following fully connected layers are then described as follow:
The first step is, FC layer that gets the output from the third convolutional layer and which exhibit neurons equal to
512 and superseded by an activation function Rectified Linear Unit(Relu) and a dropout layer.
The second step is, FC layer that gets the 512-dimensional output from the first FC layer and same procedure follow
like in the first layer.
The third step is, absolutely affiliated fully connected layer which maps to the final classes for gender
classification.At last, the output obtained from last one absolutely fully connected layer is forward to a soft-max
layer function then it assigns a probability for each label in gender detection.The anticipation is fabricated by using
the label that is having high probability from the rest of the test image used in gender recognition



### Outputs

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/faces_gender.jpg"/>



<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Gender_Classification/Assets/Title-rpng.png" height="300px" align="left"/>
<p style="clear:both;">
<h1><a name="contributing"></a><a name="community"></a> <a href="https://github.com/akshitagupta15june/Face-X">Community</a> and <a href="https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md">Contributing</a></h1>
<p>Please do! Contributions, updates, <a href="https://github.com/akshitagupta15june/Face-X/issues"></a> and <a href=" ">pull requests</a> are welcome. This project is community-built and welcomes collaboration. Contributors are expected to adhere to the <a href="https://gssoc.girlscript.tech/">GOSSC Code of Conduct</a>.
</p>
<p>
Jump into our <a href="https://discord.com/invite/Jmc97prqjb">Discord</a>! Our projects are community-built and welcome collaboration. 👍Be sure to see the <a href="https://github.com/akshitagupta15june/Face-X/blob/master/Readme.md">Face-X Community Welcome Guide</a> for a tour of resources available to you.
</p>
<p>
<i>Not sure where to start?</i> Grab an open issue with the <a href="https://github.com/akshitagupta15june/Face-X/issues">help-wanted label</a>
</p>
