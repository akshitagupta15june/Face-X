# Detective Filter


#### Requirements
- Python 3.7.x
- OpenCV
- Imutils
- Dlib library
- Download Face Landmark Detection Model (shape_predictor_68_face_landmarks.dat file) 
from [here](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat).

#### Instruction
- Clone this repository ` git clone https://github.com/akshitagupta15june/Face-X.git`
- Change Directory to ` Snapchat_Filters` then to `Detective Filter`
- Run code using the cmd ` python detective_filter.py`

### Screenshot
<img height="380" src="https://github.com/Defcon27/Face-X/blob/master/Snapchat_Filters/Detective%20Filter/assets/out.jpg">

### Detail of the Algorithm used in Detective Filter
We are using Viola Jones algorithm which is named after two computer vision researchers who proposed the method in 2001, Paul Viola and Michael Jones in their paper, “Rapid Object Detection using a Boosted Cascade of Simple Features”. Despite being an outdated framework, Viola-Jones is quite powerful, and its application has proven to be exceptionally notable in real-time face detection. This algorithm is painfully slow to train but can detect faces in real-time with impressive speed.
<br>
Given an image(this algorithm works on grayscale image), the algorithm looks at many smaller subregions and tries to find a face by looking for specific features in each subregion. It needs to check many different positions and scales because an image can contain many faces of various sizes. Viola and Jones used Haar-like features to detect faces in this algorithm.
<br>
The Viola Jones algorithm has four main steps
<br>

1.Selecting Haar-like features:<\strong>Haar-like features are digital image features used in object recognition. All human faces share some universal properties of the human face like the eyes region is darker than its neighbour pixels, and the nose region is brighter than the eye region.
<br>  
2.Creating an integral image:An integral image (also known as a summed-area table) is the name of both a data structure and an algorithm used to obtain this data structure. It is used as a quick and efficient way to calculate the sum of pixel values in an image or rectangular part of an image.

<br>
3.Running AdaBoost training: we’re training the AdaBoost to identify important features, we’re feeding it information in the form of training data and subsequently training it to learn from the information to predict. So ultimately, the algorithm is setting a minimum threshold to determine whether something can be classified as a useful feature or not.

<br>
4.Creating classifier cascades:<\strong>We set up a cascaded system in which we divide the process of identifying a face into multiple stages. In the first stage, we have a classifier which is made up of our best features, in other words, in the first stage, the subregion passes through the best features such as the feature which identifies the nose bridge or the one that identifies the eyes. In the next stages, we have all the remaining features.
