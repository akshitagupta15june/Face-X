# Smile Detection using Neural Networks

This project trains a neural network to detect smiling faces in images. 

## Process Overview

1. Dataset Preparation:
   - Organize your dataset with images of smiling faces in the `dataset/smile` folder and non-smiling faces in the `dataset/non_smile` folder.
   - Ensure test images are in a separate folder, without subfolders.
   you can download images from https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data.

2. Training the Model:
   - Run `train_model.py` to load, preprocess, and train a Convolutional Neural Network (CNN) on the dataset.
   - The model is saved as `smiling_detection_model.h5`.

3. Testing the Model:
   - Use `test_model.py` to load the saved model and predict labels for the test images.
   - The script displays test images with their predicted labels.

## Use Cases

Integrate this feature into applications like photo management systems, social media platforms, and digital marketing tools to automatically identify and tag images with smiling faces, enhancing user experience and enabling more personalized content delivery.

## Getting Started
`

1. **Install the required libraries**:

    pip install -r requirements.txt


2. **Train the model**:

    python train_model.py --dataset_path path/to/dataset --model_path smiling_detection_model.h5


3. **Test the model**:

    python test_model.py --test_path path/to/testing/dataset --model_path smiling_detection_model.h5


## License

This project is licensed under the MIT License.
