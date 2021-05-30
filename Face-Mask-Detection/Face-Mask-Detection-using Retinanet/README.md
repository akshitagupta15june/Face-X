# Face-Mask-Detection
An approach to detecting face masks in crowded places built using RetinaNet Face for face mask detection and Xception network for classification.

# The coronavirus: role of face masks in curtailing the spread of infection
Currently, we have 16 million confirmed cases and near 645,000 deaths worldwide due to the coronavirus pandemic. While some factors can be eluded as being not being under our control as individuals, we cannot say that face masks do not have a big role to play in controlling this pandemic. Upon wearing the face mask, we as the source of infection reduce the number of droplets ejected by 99 percent according to confirmed studies, and if we reduce the number of people getting infected, it also reduces the effective reproduction rate, and hence having an impact of exponential margins. Nearly half of the infected people do not show symptoms as per recent studies, which can take up to a period of 14 days to appear in an infected individual. Hence, it is really necessary for wearing masks by people in public places, and should really be made mandatory rather than being based on individual decisions, as a significant portion of people with infection lack coronavirus symptoms.

# Data Collection and Preprocessing
There were issues circumventing the data collection process, primarily the unavailability of data specific to detecting face masks as detection problem. The images did not have localized bounding boxes, and did not have category labels for the masked and without mask faces of people in the image. Due to this reason, I had to use the approach for face detection and followed by the classification of faces into mask and without mask categories.
## 1. Data Collection for Training examples
As discussed earlier, there were various issues which occurred during the collection process for training examples, due to the unavailability of datasets specific to face mask detection. A solution to the same would have been using facial landmarks to detect important indicators for faces, but this idea was not used in our implementation. Finally, it was decided that my best option would be to use a data classified into two categories of images: images of people wearing masks and the ones without mask. It is important to note that the people wearing masks on their face in an incorrect manner would be classified into the "without mask" category.
Data collection was done using a variety of sources from interfaces such as Kaggle API and Bing Search. As an additional source, the Real World Masked Face Detection (RMFD) dataset was used to facilitate training of our mask detection algorithm using the classification based approach of detecting masks followed by predicting whether the detected face contains a mask or not.
The total number of images in the training set were divided into the two categories as follows:
#### (a) Masked Images: 8072
- Source (Github: Real-world Masked Face Dataset): https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset <br> Count: 144
- Source (Bing Search API): https://api.cognitive.microsoft.com/bing/v7.0/images/search <br> Count: 1355
- Source (Kaggle Datasets): https://www.kaggle.com/koyomi455/mask-dataset <br> Count: 690
- Source (Kaggle Datasets): https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset <br> Count: 5883
#### (b) Unmasked Images: 8086
- Source (Github: Real-world Masked Face Dataset): https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset <br> Count: 1492
- Source (Kaggle Datasets): https://www.kaggle.com/koyomi455/mask-dataset <br> Count: 686
- Source (Kaggle Datasets): https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset <br> Count: 5909

## 2. Data Collection for Inference
The images and video sequence collected for the purpose of testing our model had to come from a crowded setting to provide a realistic test scenario for my proposed model for mask detection. This task was accomplished using using Google search for images and available YouTube videos for the purpose of scraping the web for realistic examples suited to inference in a real world surveillance setting.
- Source (Google Image Search): https://images.google.com/ <br> Count: 117
- Source (Associated Press Images): http://www.apimages.com/ <br> Count: 13

# Face Mask Detection
As discussed earlier, the mask detection model can be said to be a combination of classification and face detection model.
For the purpose of classification, we use transfer learning with an Xception model trained on the ImageNet dataset with a modified final fully connected layer. While using the face detection model, several different approaches were tried upon based on existing literature, and the one which worked the best was a RetinaNet Face pre-trained model which gave the highest measures of recall while experimenting on different use-cases and testing images of people in a crowded setting.
The models and implementation details for them have been discussed in an objective manner as part of this section, and while providing an insight on the approach used (and why it was chosen in the first place), we delve into our final mask detection model which was built using a combination of the classification and face detection models as were briefly described above.

### Model for Classification: Xception
A classification problem is, using available training data with defined features and class labels, building a function which can, with high levels of certainty categorize a new unseen set of data into one of the classes. Having only a limited amount of data available for training the classifier, I was inclined to use transfer learning for the purpose of our task of classifying an input image into the categories of whether the subject is wearing a mask or not. The transfer learning technique is probably one of the most revolutionary ideas to have come out in the past few years, and can be thought of reusing a pre-trained model, which is trained on another set of input images, which in our case would be the use of Xception model trained on the ImageNet database.
<br>
Implementation details for classification architecture:
- ***Batch Size***: 32
- ***Epochs***: 2
- ***Learning rate***: 1e-4 (with decay of 1e-4 / epoch)
- ***Gradient Descent Optimizer***: Adam
- ***Loss function***: Sparse categorical cross entropy
- ***Criterion for evaluation***: F1-score

### Model for Face Detection: RetinaNet Face
Face Detection is the technique of identifying human faces in digital images. While inherently a backbone to other applications, detecting a face is in fact impacted a lot in cluttered scenes, and examples of the same kind of problem can be thought of in a crowded setting, the use-case for which our mask detection algorithm is being built. For this reason, having tried various techniques from classical Computer Vision domain to using deep learning techniques, I needed to prioritize the mAP metric of detected faces in a crowded setting. Having said that, I tried techniques for face detection such as Haar cascading and MT-CNN which did not achieve a high recall. Finally, I sided with a pre-trained RetinaNet Face model, using focal loss which is able to handle the foreground-background class imbalance (an issue with one stage detectors which makes performance of single shot detectors inferior to two stage detectors for object detection) in the detected classes pretty well.

#### ***Improving the precision and recall***
I was able to improve the precision and recall of my model by a high margin in the following ways:
1. Resizing the cropped face before providing as input to the classification model.
2. For cases wherein the dimensions for height and width of the face crops fall below a threshold, increasing the dimensions of the crop by some proportion.

### Proposed Model for Face Mask Detection:
On any given test image of a crowd based setting of people, our final mask detection model runs as follows: Apply the RetinaNet Face model for face detection to generate detected face crops from the input image. Xception model for classification into mask and no-mask categories for the detected face is applied upon the detections generated by RetinaNet model. The final output of these two would be the faces detected by RetinaNet along with the predicted category for each face, that is whether the subject is wearing a mask or not.

# Results and Discussion
### 1. Results obtained on the classification model:
- Loss: 0.0182
- F1 score 0.99

### 2. Results on test images
<br>

![Crowded_Scenario 1](https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.0/blob/master/Results/Test%20Images/test_img1.png/)
![Crowded_Scenario 2](https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.0/blob/master/Results/Test%20Images/test_img2.png/)
![Crowded_Scenario 3](https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.0/blob/master/Results/Test%20Images/test_img3.png/)
![Crowded_Scenario 4](https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.0/blob/master/Results/Test%20Images/test_img4.png/)
![Crowded_Scenario 5](https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.0/blob/master/Results/Test%20Images/test_img5.png/)
![Crowded_Scenario 5](https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.0/blob/master/Results/Test%20Images/test_img6.png/)

<br>

### 3. Results on test video
![](https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.0/blob/master/Results/Test%20Video/test_video_gif.gif)

### 4. How does this approach fare in these conditions?
Using the before mentioned approach of face detection and classification of the detected face crops as wearing a mask or not works pretty well in crowded conditions. This is really important because the use cases for the regions of surveillance may include metropolitan complexes, metro stations and dense marketplaces. These conditions do not provide an ideal scenario for just any face detection algorithm, and it was really necessary that the right choice was made.
### 5. Choice of RetinaNet over MT-CNN, Haar Cascade and HOG:
While deciding upon the face detection algorithm to be used as part of my proposed solution of face detection, a pre-trained RetinaNet was chosen as the one which could, with the highest recall and precision, predict the number of faces in a crowded setting.
In an uncontrolled environment, accurate face localization remains a challenge, and I needed a model which could efficiently predict, with a very high level of certainty- the people who are not wearing a mask in a crowded setting for the mask detection algorithm to work effectively.<br>
While they work good in general settings, MT-CNN (Multi Task Cascaded Convolutional Neural Network) and classical computer vision algorithms such as Haar Cascade failed to work well in an uncontrolled environment of a crowded setting with people and in dense clusters. RetinaNet with a ResNet backbone and feature pyramid network for feature extraction works even well than some single shot detectors like Single Shot MultiBox Detector and has accuracy on par with two stage detectors like Faster RCNN, and handles foreground-background class imbalance using a modified version of Focal Loss. The class imbalance was the major issue in other single shot detectors, and helps RetinaNet have a lower loss due to easy examples while focusing on hard ones.
<br> This face detection model works well in crowded settings, which are bound to have large number of people with 'smaller' faces and with varying scale than in a general setting. On the testing examples, it can be seen that the faces detected vary in scale but are detected with a very high recall and precision.
### 6. Classification model: Choice of architecture
- Transfer learning was employed to use the model weights of the Xception model.
- Learning hyperparameters such as learning rate were chosen in an iterative manner, with recommendations taken on choices of values based on available architecture.
### 7. Criterion for evaluation: F1-score
- F1-score is the harmonic mean of precision and recall. It is chosen as the criterion for evaluation for the classification model. Being bound between 0 and 1, F1-score reaches its best value at 1 and worst at 0.
- My model achieves a high F1-score which shows that it can perform well in real world scenarios to classify with certainty, the mask and without mask categories on face crops.
