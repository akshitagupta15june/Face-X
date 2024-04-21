import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def load_model():
    model = tf.keras.models.load_model('Awesome-face-operations/Gender_Classification/Gender_final.h5')
    return model

model = load_model()

class_labels = {0: 'Male', 1: 'Female'}

def preprocess_image(image):
    img = np.array(image.resize((32, 32)))
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img / 255.0
    return img

def predict(image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

if __name__ == '__main__':
    st.title('Gender Classification')
    st.write('Created by [Avdhesh Varshney](https://github.com/Avdhesh-Varshney)')
    name = st.text_input('Enter your name:')
    st.write('Upload an image for gender classification')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and name != '':
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            try:
                prediction = predict(image)
                predicted_class = np.argmax(prediction)
                st.success(f'Predicted Gender: {class_labels[predicted_class]}', icon='✅')
                st.balloons()
            except:
                st.error('There is an error which predicting the gender classification!', icon='🚨')

