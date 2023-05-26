# Emoji-based-Emotion-prediction

In this directory, we have added a feature to predict emotions using images of various emojis depicting sentiments and feelings. The model we utilised to train in this project is *MobileNetV2* where we used transfer learning to acquire the desired results.

## Dependencies:
	1. python3.8 or greater version
	2. Tensorflow2.8
	3. numpy

Steps to be followed for initially begin the execution of files:
1. Unzip *data_and_model.zip*
2. run the command in the terminal *python predict.py*


## Dataset Details:
	We downloaded 168 three kinds of images from internet of emojis and divided them into three labels happy, sad and angry along with a directory data which contains three directories with the class label's name containing respective images.
	
## Prediction result of our model:
![2dd5bc89-15eb-4a9e-bf2e-bf54d296817b](https://user-images.githubusercontent.com/86379589/196512758-3af2ae92-ba30-4ea6-a4c7-384613c28f73.png)


## Training a custom model:
If we want to train a custom model, we need to store images in the form of folders within the data directory where each folder will contain images with their respective names or labels. After this we need to run the provided .ipynb file on jupyter notebook with the name *emoji_emotion_recog_jupyter.ipynb* it will train the model and save it into the model folder.
