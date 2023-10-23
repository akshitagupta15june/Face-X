# Facial Recongtion using Ensemble Learning.

## What is deepface?
Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python. It is a hybrid face recognition framework wrapping state-of-the-art models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib. The library is mainly based on Keras and TensorFlow.

Deepface also offers facial attribute analysis including age, gender, facial expression (including angry, fear, neutral, sad, disgust, happy and surprise) and race (including asian, white, middle eastern, indian, latino and black) predictions. Analysis function under the DeepFace interface is used to find demography of a face.

## First we have to install the deepface library.
The easiest way to install deepface is to download it from [`PyPI`].
```
pip install deepface
```
## Secondly we should install the dependencies.
```
pip install tensorflow==2.4.1
pip install keras==2.4.3
```
## Face Recognition

A modern face recognition pipeline consists of 4 common stages: detect, align, represent and verify. Deepface handles all these common stages in the background. You can just call its verification, find or analysis function in its interface with a single line of code.

Face Verification - [`Demo`]

Verification function under the deepface interface offers to verify face pairs as same person or different persons. You should pass face pairs as array instead of calling verify function in a for loop for the best practice. This will speed the function up dramatically and reduce the allocated memory.

this is the sample code we can use to verify different faces
```
from deepface import DeepFace
result  = DeepFace.verify("img1.jpg", "img2.jpg")
#results = DeepFace.verify([['img1.jpg', 'img2.jpg'], ['img1.jpg', 'img3.jpg']])
print("Is verified: ", result["verified"])
```
## Ensemble learning for face recognition 

A face recognition task can be handled by several models and similarity metrics. Herein, deepface offers a special boosting and combination solution to improve the accuracy of a face recognition task. This provides a huge improvement on accuracy metrics. Human beings could have 97.53% score for face recognition tasks whereas this ensemble method passes the human level accuracy and gets 98.57% accuracy. On the other hand, this runs much slower than single models.

For comparing two photos we can also use this code.
```
obj=DeepFace.verify("Dataset\steve1.jpg","Dataset\steve2.jfif" , model_name="Ensemble") 
```
Here we insert two photos and use the ensemble model.

## For checking whether a given face is in a database.
```
df= DeepFace.find(img_path="Database\mark1.jpg",db_path="Database",model_name="Ensemble")
```
where img_path is the image you want to find resemblance, db_path is the database folder and model_name is the ensemble model.
