# Facial Expression Recognition
A CNN model to recognize facial expressions from 2D images.

### Dataset
FER2013 - https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge  
The data consists of 48x48 pixel grayscale images of faces. The emotion shown in the facial expression is of one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).  


The images are divided in the following categories-  
1. Training - 28709
2. Private Test - 3589
3. Public Test - 3589

### Requirements  
1. Python 3.x
2. Google Colab

### Model Layers
- Conv2D
- MaxPooling2D
- Batch Normalization
- Dense
- Flatten
- Dropout  
in different numbers and order

### Accuracy and loss
- loss: 0.8406 
- accuracy: 0.6644 
- val_loss: 1.5589 
- val_accuracy: 0.6403

### Files usage
- face-exp.ipynb : Data preprocessing, the CNN model and different approaches to it.
- weights.h5 : saved weights after training with maximum accuracy
- model.json : saved model configuration
- predict.py : script to predict emotions in real time from camera feed.
