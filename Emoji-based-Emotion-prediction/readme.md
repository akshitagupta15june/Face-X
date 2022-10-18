# Emoji-based-Emotion-prediction

In this directory, we added a feature to predict emotion with the emoji images. The model which we trained here is *MobileNetV2*, we used transfer learning to accomplish desired results.

## Dependencies:
	1. python3.8 or greater version
	2. Tensorflow2.8
	3. numpy

To run the things here we need to follow few steps:
1. Unzip *data_and_model.zip*
2. run the command in the terminal *python predict.py*


## Dataset Details:
	We downloaded 168 three kinds of images from internet of emojis and we divided them into three label smiling, sad, and angry the 
	there is a directory data which contains three directory with the class label's name and each contains respective images.
	
## Prediction result of our model:
![2dd5bc89-15eb-4a9e-bf2e-bf54d296817b](https://user-images.githubusercontent.com/86379589/196512758-3af2ae92-ba30-4ea6-a4c7-384613c28f73.png)


## Training a custom model:
 If we want to train a custom model then we just need to put images foled into the data directory, where each folder will contain images with their respective names. Afer this we need to run the provided jupyter notebook with the name *emoji_emotion_recog_jupyter.ipynb* it will train the model and save it into the model folder.
