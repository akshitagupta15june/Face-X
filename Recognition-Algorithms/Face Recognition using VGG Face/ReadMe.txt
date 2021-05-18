# Face Recognition with VGG_Face-net transfer learning.
### Step 1: Dataset acquisition
##### Train data
Gathered 50 images of 5 most powerful world leaders Trump,Putin,Jinping,Merkel and Modi of 10 images each.
Also my 10 images.
##### Test data
18 images with 3 images each of above 6 persons including me.

### Step 2: Detect faces.
Using `dlib cnn face detector` find faces in image and crop faces and store them in separate folders sorting by individual person. <br>
Download 'mmod_human_face_detector' to use as 'dlib cnn face detector',br>
```
### Step 3: Download and load VGG_face weights.
As vgg-face weights are not available as `.h5` file to download,from this 
[article](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)<br>
Download Vgg-face weights from google drive with 
```
! gdown https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo
```
Define vgg-face model architecture in tensorflow-keras and load weights.

### Step 4: Get embeddings for faces.
To get embeddings from Vgg-face net remove last softmax layer and it outputs `2622` units in last flatten layer which is our
embeddings for each face.<br>
These embeddings are used later to train a softmax regressor to classify the person in image.

### Step 5: Train Softmax regressor for 6 person classification from embeddings.
Prepare train data and test data from those `2622` embeddings and feed into a simple softmax regressor with `3` layers containing first layer with `100` units and `tanh` activation function , second layer with `10` units and `tanh` activation function and third layer with `6` units for each person with `softmax` activation.

### Predictions
