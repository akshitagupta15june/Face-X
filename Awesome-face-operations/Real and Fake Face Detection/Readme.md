<h1>PROJECT TITLE</h1>
<br>
<h3>
<u>Real and Fake Face Detection</u></h3>
<br>
<h1>REQUIREMENTS</h1>
<ol>
<li>keras -- 3.7</li>
<li>numpy -- 1.25</li>
<li>pandas --1.3.5</li>
<li>PIL --9.5</li>
<li>cv2 --4.7</li>
<li>matplotlib --3.6</li>
<li>tensorflow --2.12</li>
</ol>
<br>
<h1>LIBRARIES NEEDED</h1>
<br>
The following libraries are essential :-
<br>
<ol>
<li>Keras - open source python library which facilitates deep neural networks like convolutional ,recurrent and their combinations</li>
<li>Tensorflow - open source library which supports Machine Learning powered models and applications.It also works with Keras and is of high utility </li>
<li>Scikit-learn - Simple and efficient tools for data mining and data analysis. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means.
</li>
<br>

<h1>SET UP</h1>
<br>
The code is done on google colab notebooks which can be runned directly.Installations of necessary libraries and API mentioned in the Requirements Section are compulsory for the execution of the code.
The dataset needs to be downloaded from the below given link and the path for the same needs to be changed according to the location where the dataset gets downloaded on your machine.<br>
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
<h1>BRIEF DESCRIPTION OF MODELS AND THE RESULTS OBTAINED FROM THEM</h1>
<br>
<h1>1. MOBILENETV2 MODEL</h1>
<br>

![mobilenet conv blocks](https://github.com/kanishkakataria/Images/assets/85161519/142333e3-efcf-4386-8bfd-eede3aa842d2)<br>
Depthwise Separable Convolution is introduced which dramatically reduce the complexity cost and model size of the network, which is suitable to Mobile devices, or any devices with low computational power. In MobileNetV2, a better module is introduced with inverted residual structure. Non-linearities in narrow layers are removed this time. With MobileNetV2 as backbone for feature extraction, state-of-the-art performances are also achieved for object detection and semantic segmentation. 

<br>
<ol>
<li>In MobileNetV2, there are two types of blocks. One is residual block with stride of 1. Another one is block with stride of 2 for downsizing.</li>
<li>There are 3 layers for both types of blocks.</li>
<li>This time, the first layer is 1×1 convolution with ReLU6.</li>
<li>The second layer is the depthwise convolution.</li>
<li>The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if ReLU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.</li>
<li>The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if ReLU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.</li>
</ol>

<h2>Confusion Matrix</h2>

![confusion matrix](https://github.com/kanishkakataria/Images/assets/85161519/d55ce591-3733-4281-bfdc-b246e0fe8d0a)<br>
<h2>Accuracy Curve</h2>

![acc_curve](https://github.com/kanishkakataria/Images/assets/85161519/308d7b80-bd5e-4c02-afaf-4132516b6992)<br>
<h2>Loss Curve</h2>

![loss_curve](https://github.com/kanishkakataria/Images/assets/85161519/0b6ce9aa-8251-489a-a0bf-6f088fb566c3)
<br>

<h1>2. VGG16 MODEL</h1>
<br>

![model architecture](https://github.com/kanishkakataria/Images/assets/85161519/e8c87859-695f-448e-bb54-1bb78a6c3744)
<br>
A convolutional neural network is also known as a ConvNet, which is a kind of artificial neural network. A convolutional neural network has an input layer, an output layer, and various hidden layers. VGG16 is a type of CNN (Convolutional Neural Network) that is considered to be one of the best computer vision models to date. The creators of this model evaluated the networks and increased the depth using an architecture with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations. They pushed the depth to 16–19 weight layers making it approx — 138 trainable parameters.

<h2>USAGE</h2>

VGG16 is object detection and classification algorithm which is able to classify 1000 images of 1000 different categories with 92.7% accuracy. It is one of the popular algorithms for image classification and is easy to use with transfer learning.

<h1>ARCHITECTURE</h1>
<ol>
<li>The 16 in VGG16 refers to 16 layers that have weights. In VGG16 there are thirteen convolutional layers, five Max Pooling layers, and three Dense layers which sum up to 21 layers but it has only sixteen weight layers i.e., learnable parameters layer.
</li>
<li>VGG16 takes input tensor size as 224, 244 with 3 RGB channel</li>
<li>Most unique thing about VGG16 is that instead of having a large number of hyper-parameters they focused on having convolution layers of 3x3 filter with stride 1 and always used the same padding and maxpool layer of 2x2 filter of stride 2.</li>
<li>The convolution and max pool layers are consistently arranged throughout the whole architecture</li>
<li>Conv-1 Layer has 64 number of filters, Conv-2 has 128 filters, Conv-3 has 256 filters, Conv 4 and Conv 5 has 512 filters.
Three Fully-Connected (FC) layers follow a stack of convolutional layers: the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer.</li>
</ol>
<br>

<h2>CONFUSION MATRIX</h2>

![confusion matrix](https://github.com/kanishkakataria/Images/assets/85161519/e0202a79-53d4-4a9e-96cc-f7f65dfc0cb9)<br>
<h2>ACCURACY CURVE</h2>

![acc_curve](https://github.com/kanishkakataria/Images/assets/85161519/cb763b6a-d816-4231-aa14-b238c5980690)<br>
<h2>LOSS CURVE</h1>

![loss_curve](https://github.com/kanishkakataria/Images/assets/85161519/5ccd3909-70c7-4008-bf9a-ec9413d8066a)<br>

<h1>3. INCEPTIONV3 MODEL</h1>
<br>

![model architecture](https://github.com/kanishkakataria/Images/assets/85161519/e697b22d-0fa8-42b9-a5ba-2fe482bad7fb)<br>
The Inception V3 is a deep learning model based on Convolutional Neural Networks, which is used for image classification. The inception V3 is a superior version of the basic model Inception V1 which was introduced as GoogLeNet in 2014. As the name suggests it was developed by a team at Google.
<br>
The inception v3 model was released in the year 2015, it has a total of 42 layers and a lower error rate than its predecessors. Let's look at what are the different optimizations that make the inception V3 model better.
<br>
<h4>The major modifications done on the Inception V3 model are:-</h4>
<ol>
<li>Factorization into Smaller Convolutions
</li> 
<li>Spatial Factorization into Asymmetric Convolutions</li>
<li>Utility of Auxiliary Classifiers</li>
<li>Efficient Grid Size Reduction</li>
</ol>
<br>
The inception V3 is just the advanced and optimized version of the inception V1 model. The Inception V3 model used several techniques for optimizing the network for better model adaptation.
<br>
<ol>
<li>It has higher efficiency</li>
<li>It has a deeper network compared to the Inception V1 and V2 models, but its speed isn't compromised.</li>
<li>It is computationally less expensive.</li>
<li>It uses auxiliary Classifiers as regularizes.</li>
</ol>
<br>

<h2>CONFUSION MATRIX</h2>

![confusion matrix](https://github.com/kanishkakataria/Images/assets/85161519/8e60f902-17fa-4e84-99e7-e778f21b3446)<br>
<h2>ACCURACY CURVE</h2>

![acc_curve](https://github.com/kanishkakataria/Images/assets/85161519/e1a3e08a-8e78-40bc-8076-066b0e170f38)
<br>
<h2>LOSS CURVE</h2>

![loss_curve](https://github.com/kanishkakataria/Images/assets/85161519/f5925ea7-87cb-4e00-9d4a-b65e10eef5e6)<br>
<h2>XCEPTION MODEL</h2>
<br>

![Modified Deptthwise Separable Convolution in Xception](https://github.com/kanishkakataria/Images/assets/85161519/84790e95-81e7-495e-9ba8-f96d7391330c)
<br>
Xception is a deep convolutional neural network architecture that involves Depthwise Separable Convolutions. It was developed by Google researchers. Google presented an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads them to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.<br>
The modified depthwise separable convolution is the pointwise convolution followed by a depthwise convolution. This modification is motivated by the inception module in Inception-v3 that 1×1 convolution is done first before any n×n spatial convolutions. Thus, it is a bit different from the original one. (n=3 here since 3×3 spatial convolutions are used in Inception-v3.)
<br>
<h2>Two minor differences:</h2>
<br>
1. The order of operations: As mentioned, the original depthwise separable convolutions as usually implemented (e.g. in TensorFlow) perform first channel-wise spatial convolution and then perform 1×1 convolution whereas the modified depthwise separable convolution perform 1×1 convolution first then channel-wise spatial convolution. This is claimed to be unimportant because when it is used in stacked setting, there are only small differences appeared at the beginning and at the end of all the chained inception modules.
<br>
2. The Presence/Absence of Non-Linearity: In the original Inception Module, there is non-linearity after first operation. In Xception, the modified depthwise separable convolution, there is NO intermediate ReLU non-linearity.


<h2>CONFUSION MATRIX</h2>

![confusion matrix_Xception](https://github.com/kanishkakataria/Images/assets/85161519/4c9b6600-9a20-4a09-8f50-356b524315e5)<br>
<h2>ACCURACY CURVE</h2>

![Acc_curve](https://github.com/kanishkakataria/Images/assets/85161519/4b43ac5e-d89e-4941-acd8-e67c37175e09)<br>
<h2>LOSS CURVE</h2>

![loss_Curve](https://github.com/kanishkakataria/Images/assets/85161519/7a5d1c76-1118-4881-8b0b-aeec9bec6e33)
<br>

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

<h1>REFERENCES</h1>
<ol>
<li>St, S., Ayoobkhan, M. U. A., Kumar, K., V., Bacanin, N., Bacanin, N., Štěpán, H., & Pavel, T. (2022, February 22). Deep learning model for deep fake face recognition and detection. PeerJ. https://doi.org/10.7717/peerj-cs.881</li>
<li>Kvlkeerthi. (2019). Starter: Real and Fake Face Detection b7a8f32c-f. Kaggle. https://www.kaggle.com/code/kvlkeerthi250299/starter-real-and-fake-face-detection-b7a8f32c-f</li>
<li>Rage. (2023). Facial Landmark Detection+CNN. Kaggle. https://www.kaggle.com/code/rage147/facial-landmark-detection-cnn</li>
<li>Salman, F. M. (2022). Classification of Real and Fake Human Faces Using Deep Learning. https://philpapers.org/rec/SALCOR-3</li>
<li>IRJMETS - International Research Journal of Modernization in Engineering Technology and Science. (n.d.). IRJMETS. IRJMETS International Research Journal of Modernization in Engineering Technology and Science. https://www.irjmets.com/uploadedfiles/paper/volume3/issue_3_march_2021/7290/1628083311</li>
</ol>
<br>
<h1>CONTRIBUTOR</h1>

NAME - KANISHKA KATARIA
<br>
GITHUB - https://github.com/kanishkakataria
<br>
LINKEDIN - https://kanishkakataria.vercel.app/
<br>
PORTFOLIO -https://kanishkakataria.vercel.app/
