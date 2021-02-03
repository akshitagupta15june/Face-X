### Demonstration of different algorithms and operations on faces

There are several approaches for recognizing a face. The algorithm can use statistics, try to find a pattern which represents a specific person or use a convolutional neural network. 
<div align="center">
<img src="https://media.giphy.com/media/AXorq76Tg3Vte/giphy.gif" width="20%"><br>
</div>

The algorithms used for the tests are Eigenfaces, Fisherfacesand local binary patterns histograms which all come from the library OpenCV. Eigenfaces and Fisher faces are used with a Euclidean distance to predict the person. The algorithm which is using a deep convolutional neural network is the project called OpenFace.

This can be used for automatic face detection attendance system in recent technology.



`
Recognition of faces by different algorithms and frameworks. Despite a variety of open-source face recognition frameworks available, there was 
no ready-made solution to implement. So In this project all kind of algorithms are implemented and even with various operations that can be implemented
in a frontal face. The available algorithms processed only high-resolution static shots and performed insufficiently.
`


### Requirements 👇
- Python3.6+
- virtualenv (`pip install virtualenv`)

### Installation 🖥
- `virtualenvv env`
- `source venv/bin/activate` (Linux)
- `venv\Scripts\activate` (Windows)
- `pip install -r requirements.txt`
- Create an .env file, copy the content from .env.sample and add your data path. Example: `DATA_PATH = "./foto_reco/"`


## Comparative Study of the Algorithms used here :

This project holds different type of deep learning models on different frameworks. every single model has it's uniqueness and contribute vastly to the deep learning domain .
If we try to compare them, we might find better understanding over those and this would be great for all of us :)

Model | Creator | Published | Misc 
--- | --- | --- | --- 
LBPH | C. Silva | March,2015 | Got the highest accuracy in all experiments, but this algorithm has the higher impact of the negative light exposure and high noise level more than the others that are statistical approach.
LBP_SVM | C. Silva | March,2015 | The accuracy is reported at 90.52% using SVM which has a gamma value of 0.0000015 and penalty parameter of the error term C = 2.5 while using the RBF kernel.
MobileNetV2 | Google AI | April,2018 | Faster for the same accuracy across the entire latency spectrum. In particular, the new models use 2x fewer operations, need 30% fewer parameters and are about 30-40% faster on a Google Pixel phone than MobileNetV1 models, all while achieving higher accuracy.
EffecientNet | Google AI | May, 2019 | On the ImageNet challenge, with a 66M parameter calculation load, EfficientNet reached 84.4% accuracy and took its place among the state-of-the-art.
EigenFaceRecogniser | M. Turk and A. Pentland | 1991 | The accuracy of Eigenface is satisfactory (over 90 %) with frontal faces. Eigenface uses PCA. A drawback is that it is very sensitive for lightening conditions and the position of the head. Fisherface is similar to Eigenface but with improvement in better classification of different classes image.
FisherFaceRecogniser | Aleix Martinez | 2011 | Fisherface is a technique similar to Eigenfaces but it is geared to improve clustering of classes.  While Eigenfaces relies on PCA, Fischer faces relies on LDA (aka Fischer’s LDA) for dimensionality reduction.
GhostNet | Huwayei Noah | Recent | GhostNet can achieve higher recognition performance (75% , top-1 accuracy) than MobileNetV3 with similar computational cost on the ImageNet ILSVRC-2012 classification dataset.
KNN | Evelyn Fix and Joseph Hodges | 1951 | K-Nearest Neighbor face recognition delivered best accuracy 91.5% on k=1. KNN showed the faster execution time compared with PCA and LDA. Time execution of KNN to recognize face was 0.152 seconds on high-processor. Face detection and recognition only need 2.66 second to recognize on low-power ARM11 based system
ResNet-50 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | Dec,2015 | ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database [1]. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.
DenseNet121 | Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger | Jan,2018 | Got Best Paper Award with over 2000 citations. It is jointly invented by Cornwell University, Tsinghua University and Facebook AI Research (FAIR)
VGG-19 | Karen Simonyan, Andrew Zisserman | April,2015 | This model achieves 75.2% top-1 and 92.5% top-5 accuracy on the ImageNet Large Scale Visual Recognition Challenge 2012 dataset.
MTCNN | Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao | April,2016 | One of the hottest model used most widely recently for its high precision and outstanding real time performance among the state-of-art algorithms for face detection. Then, the first basic application of portrait classification is researched based on MTCNN and FaceNet. Its direction is one of the most classical and popular area in nowadays AI visual research, and is also the base of many other industrial branches.

![](https://miro.medium.com/max/1416/0*6wtXZPL89Apg2rlH) 

![](https://imgs.developpaper.com/imgs/1527989268-5de6f88a07966_articlex.png)


We can see that the models are very new in this world and also can see the quick evolution of them .Face Cecognition with Deep learning has now become **PIECE OF CAKE** with the help of this Benchmark Models. 
