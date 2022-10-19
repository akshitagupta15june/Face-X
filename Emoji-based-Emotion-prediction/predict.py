import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


with tf.device('/CPU:0'):
    saved_model_path = "data_and_model/model/Model.hdf5"
    img_path = 'data_and_model/test_data/images (32).jpg'
    train_data_path = "data_and_model/data/"
    img_size = 224

    labels = os.listdir(train_data_path)

    loaded_model = load_model(saved_model_path)

    img = load_img(img_path, target_size = (img_size, img_size))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    plt.imshow(load_img(img_path, target_size = (img_size, img_size)))

    plt.title(labels[np.argmax(loaded_model.predict(img))])
    plt.savefig("output_Prediction.png")
    plt.show()