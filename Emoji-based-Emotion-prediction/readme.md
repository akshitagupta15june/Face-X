# Emoji-based-Emotion-prediction

## Description
This project uses emoji photos to predict their sentiments. To achieve the intended outcome, transfer learning was employed to train the *MobileNetV2* model.

## Dependencies
1. python3.8 or greater version
2. Tensorflow2.8
3. numpy
4. matplotlib

## Dataset Details
The data collection contains 168 different categories of images. They are classified as smiling, sad, or angry. A directory data contains three directories with the class label's name. These labels include the images that correspond to them.

## Working
* The dependencies are imported.
* The data generators are setup for training and validation data.
* The *ImageDataGenerator* is configured with various data augmentation techniques such as rotation, width and height shift, and horizontal flip. It divides the data into subsets for training and validation.
* A model is constructed by combining the *MobileNetV2* base model with additional layers for customization.
* The model is trained to predict the label.

## Training a Custom Model
To train a custom model put the images folder into the data directory where each folder will contain images with their respective names. After this run the provided jupyter notebook, *emoji_emotion_recog_jupyter.ipynb*, the model will be trained and saved into the model folder.

## Getting Started
* Clone the repository in your terminal
  ` git clone https://github.com/akshitagupta15june/Face-X.git `
* Switch directory
  ` cd Emoji-based-Emotion-prediction `
* Run file
  ` emoji_emotion_recog_jupyter.ipynb `

## Want to Contribute?
If you wish to contribute refer CONTRIBUTING.md:
` https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md `
	
## Result
![2dd5bc89-15eb-4a9e-bf2e-bf54d296817b](https://user-images.githubusercontent.com/86379589/196512758-3af2ae92-ba30-4ea6-a4c7-384613c28f73.png)

