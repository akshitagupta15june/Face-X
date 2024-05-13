import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging

def predict_emotion(img_path, saved_model_path, train_data_path, img_size):
    try:
        # Load the saved model
        loaded_model = load_model(saved_model_path)

        # Load image and preprocess it
        img = load_img(img_path, target_size=(img_size, img_size))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the emotion
        predictions = loaded_model.predict(img_array)[0]
        labels = os.listdir(train_data_path)

        # Display the predicted emotion and probabilities
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Input Image')
        
        plt.subplot(1, 2, 2)
        plt.bar(labels, predictions)
        plt.title('Emotion Probabilities')
        plt.xlabel('Emotions')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()

        predicted_label = labels[np.argmax(predictions)]
        plt.suptitle(f'Predicted Emotion: {predicted_label}')

        # Save the prediction visualization
        plt.savefig("output_Prediction.png")
        plt.show()

        logging.info("Prediction successful")
    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    saved_model_path = "data_and_model/model/Model.hdf5"
    img_path = 'data_and_model/test_data/images (32).jpg'
    train_data_path = "data_and_model/data/"
    img_size = 224

    predict_emotion(img_path, saved_model_path, train_data_path, img_size)
