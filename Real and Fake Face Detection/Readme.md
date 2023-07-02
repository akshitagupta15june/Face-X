<h1>PROJECT TITLE</h1>
<br>
<h3>
<u>Real and Fake Face Detection</u></h3>
<br>

<h1>DATASET</h1>
<br>
The Dataset is taken from Kaggle.It comprises of two folders named Real and Fake having 700 Fake and 589 Real Images of Human Face.Moreover, the fake faces collected in this dataset are generated using the StyleGAN2, which present a harder challenge to classify them correctly even for the human eye. The real human faces in this dataset are gathered so that we have a fair representation of different features(Age, Sex, Makeup, Ethnicity, etc…) that may be encountered in a production setup.
<br>
Dataset Link :- https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces
<br>

<h1>GOAL</h1>
<br>
The project uses 500 images of Real and 500 images of Fake faces.The aim is to classify the face image as Real and Fake through binary classification.<br>
The models used are Transfer Learning models which are pretrained on the imagenet weights.Baseline model of each algorithm along with customised convolutional,dense and dropout layers is constructed for training of the model.
<br>
Real Fake Face Detection has usage in the area of surveillance for military and industrial applications.For behaviour cloning and automation purposes.
<br>

<h1>DESCRIPTION</h1>

The project aims at classifying the images of the staircases as ascending and descending by using Transfer Learning Models which are pretrained on imagenet weights.<br>
<br>
<h2>METHODOLOGY</h2>
<br>
<ol>
<li>The necessary libraries are imported and installed.</li><li>An array of these 1000 samples concatenated with label(Real and Fake) is formed into a dataset which will be used for training and testing of the model.</li><li>The dataset is split into training and testing using train_test_split function of sklearn.The train size is kept 0.7 for all the models.</li><li>Label Encoder is used to encode the labels into 0 and 1</li><li>Normalization is done to normzalise the pixel values between 0 and 1</li>
<li>Four Transfer Learning models namely MobileNetV2,VGG16,XCeption,InceptionV3 are used for the training purpose.Sequential models are created of baseline transfer learning models along with convoluted layer,dense layers.Dropout layer is used to avoid overfitting of the model.</li>
<li>Loss Function used is Binary Crossentropy,optimizer is Adam,Metrics is Accuracy</li>
<li>Models are trained on number of epochs value ranging from 25 to 50.</li>
<li>To analyse the accuracy and loss Confusion Matrix,Classification Report,Accuracy Graph (Training Accuracy VS Validation Accuracy),Loss Graph(Training Accuracy VS Validation Accuracy) are generated.
</li>
<li>Some test samples are used to test on the trained models and predictions are made.
</li>
</ol>
<br>
<h1>MODELS USED</h1>
<br>

Transfer Learning Models pretrained on imagenet weights are used for training.Four Models namely MobileNetV2,VGG16,XCeption,InceptionV3 are used.These models are robust and have high learning rate as they are been already trained on similiar data that is imagenet.So they give better accuracy and training.
For each model different number of convolutional,dense and dropout layers are used as per their training accuracy.
Every model is different in its pursuit and are easy trainable.
<br>


<h1>LIBRARIES NEEDED</h1>

The following libraries are essential :-
<br>
<ol>
<li>Keras - open source python library which facilitates deep neural networks like convolutional ,recurrent and their combinations</li>
<li>Tensorflow - open source library which supports Machine Learning powered models and applications.It also works with Keras and is of high utility </li>
<li>Scikit-learn - Simple and efficient tools for data mining and data analysis. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means.
</li>


<h1>ACCURACIES</h1>
<ol>
<li>MOBILENETV2 - 99.8%</li>
<li>VGG16 - 99.5%</li>
<li>INCEPTIONV3 - 99.8%</li>
<li>XCEPTION - 99.9%</li>

</ol>

<h1>CONCLUSION</h1>

The models are trained well gaining validation accuracies greater than 99%.
To avoid overfitting,dropout layers was used.The number of layers,train test split and batch size was changed to perform trial and testing for better accuracies.

VGG16 model is found to be the best as it is not overfitting and generalises the data well comparatively.

<h1>CONTRIBUTOR</h1>

NAME - KANISHKA KATARIA
<br>
GITHUB - https://github.com/kanishkakataria
<br>
LINKEDIN - https://kanishkakataria.vercel.app/
<br>
PORTFOLIO -https://kanishkakataria.vercel.app/
