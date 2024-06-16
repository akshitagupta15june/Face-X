from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

params_path = 'params.yaml'
params = read_yaml('params.yaml')
current_dir = os.path.abspath(os.path.dirname(__file__))

# Construct path to config.yaml using os.path.join
config_path = os.path.join(current_dir, 'config', 'config.yaml')
print("Resolved config path:", config_path)

# Read YAML file
config = read_yaml(config_path)
artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload
upload_image_dir = artifacts['upload_image_dir']
uploadn_path = os.path.join(artifacts_dir, upload_image_dir)

# pickle_format_data_dir
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

#Feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)


#params_path
model_name = params['base']['BASE_MODEL']
include_tops = params['base']['include_top']
input_shapes = params['base']['input_shape']
poolings = params['base']['pooling']



detector = MTCNN()
model = VGGFace(model=model_name,include_top=include_tops,input_shape=(224,224,3),pooling=poolings)
feature_list = pickle.load(open(features_name,'rb'))
filenames = pickle.load(open(pickle_file,'rb'))

# save_uploaded_image
def save_uploaded_image(uploaded_image):
    try:
        create_directory(dirs=[uploadn_path])

        with open(os.path.join(uploadn_path,uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


# extract_features
def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


# recommend image
def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


# streamlit
#st.title('Which Bollywood Celebrity You look like?')
st.title('To whom does your face match?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join(uploadn_path,uploaded_image.name),model,detector)
        # recommend
        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # display
        col1,col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos],width=300)
