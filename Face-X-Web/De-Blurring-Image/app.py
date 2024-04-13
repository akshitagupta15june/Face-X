from tensorflow import saved_model
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import model_from_json, load_model
import io

def build_model(model_architecture_path, model_weights_path):
    with open(model_architecture_path, "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_path)
    return model

def preprocess_image(image):
    resized_image = image.resize((64, 64))
    processed_image = np.array(resized_image) / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def deblur_image(image, model):
    processed_image = preprocess_image(image)
    output_image = model.predict(processed_image)
    output_image = (output_image * 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_image[0])
    return output_image

def deblur_app(model):
    st.title("Image Deblurring App")

    name = st.text_input("Enter your name")
    age = st.number_input("Enter your age", min_value=12, step=1)

    uploaded_file = st.file_uploader("Upload an image for deblurring", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            if st.button("Process Image"):
                st.subheader("Result")
                st.write(f"Hello {name}, here is your processed image:")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                with col2:
                    output_image = deblur_image(image, model)
                    st.image(output_image, caption="Processed Image", use_column_width=True)
                    st.balloons()

                    output_bytes = io.BytesIO()
                    output_image.save(output_bytes, format='PNG')
                    output_bytes.seek(0)
                    st.download_button(label='Download Output Image', data=output_bytes, file_name='output_image.png', mime='image/png')
        except:
            st.write("There is an error in the uploaded content!")

if __name__ == "__main__":
    model_deblur_architecture_path = 'Face-X-Web/De-Blurring-Image/de-blur-cnn-model/deblur_CNN_architecture.json'
    model_deblur_weights_path = 'Face-X-Web/De-Blurring-Image/de-blur-cnn-model/deblur_CNN.weights.h5'
    model_deblur_keras_path = 'Face-X-Web/De-Blurring-Image/de-blur-cnn-model/deblur_CNN_model.keras'

    model_autoencoder_architecture_path = 'Face-X-Web/De-Blurring-Image/auto-encoder-model/autoencoder_model_architecture.json'
    model_autoencoder_weights_path = 'Face-X-Web/De-Blurring-Image/auto-encoder-model/autoencoder_model.weights.h5'
    model_autoencoder_keras_path = 'Face-X-Web/De-Blurring-Image/auto-encoder-model/autoencoder_model.keras'

    try:
        # Method - 1
        # model = build_model(model_deblur_architecture_path, model_deblur_weights_path)
        # model = build_model(model_autoencoder_architecture_path, model_autoencoder_weights_path)

        # Method - 2
        model = load_model(model_deblur_keras_path)
        # model = load_model(model_autoencoder_keras_path)

        # Method - 3
        # model = saved_model.load('Face-X-Web/De-Blurring-Image/auto-encoder-model/autoencoder')
        # model = saved_model.load('Face-X-Web/De-Blurring-Image/de-blur-cnn-model/deblur_CNN_model')

        deblur_app(model)

    except:
        st.write('There is an error in model building and loading!')

