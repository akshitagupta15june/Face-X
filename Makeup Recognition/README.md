# Makeup Detection with CNN

This repository contains code for a Convolutional Neural Network (CNN) model that performs makeup detection. The model is trained to classify images into three categories: "No Makeup," "Natural Makeup," and "Bold Makeup." The code uses the TensorFlow library to build, train, and evaluate the CNN model.

## Dependencies

The following dependencies are required to run the code:

- TensorFlow
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The code expects the dataset to be organized in a specific structure within the "Makeup_Detection" directory. The dataset directory should contain subdirectories for each class, where the images corresponding to that class are stored. Ensure that the images are properly labeled and categorized.
The two subdirectories are - "MakeUp" and "No Makeup"


Make sure to modify the code to adjust any necessary parameters or file paths according to your specific setup.

## Working

1. The code loads the dataset using the `tf.keras.preprocessing.image_dataset_from_directory` function. It splits the dataset into training and testing sets.

2. The CNN model is built using the `tf.keras.Sequential` API. It consists of convolutional layers with max pooling, followed by flattening and dense layers. The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

3. The training process involves iterating over the training dataset for a specified number of epochs, here the epochs are 20. The model's performance is evaluated on the validation dataset after each epoch.

4. After training, the model is evaluated on the test dataset using the `evaluate` method. The evaluation metrics, such as accuracy, are displayed.

5. Finally, predictions are made on the test set using the trained model.

## Results

The performance of the model can be assessed by examining the evaluation metrics and the predicted outputs. The results analyzed is as follows:

**Accuracy - 88.46 %**






