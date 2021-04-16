Iris Recognition
================

Iris recognition using applied machine learning on CASIA iris images
dataset

## IrisLocalization.py
===================

The function IrisLocalization(images) does the following: 

1. It uses a Bilateral filter to remove the noise by blurring the image 

2. We project the image coordinates in the horizontal and vertical directions, and find the minimum(as the minimum would be a dark region of the pupil) to find the approximate center of the pupil. 

3. We next use this approximate center to binarize a 120 x 120 space around the pupil, as defined in the paper, to re-estimate the pupil center. 

4. We perform Canny edge detection on a masked image to get only a few edges around the pupil. If the image was not masked we would get a lot of edges of the eyelashes,etc. 

5. We then implement Hough transformation on the edged image to detect all possible circles in the image. We take the Hough circle that has center closest to the pupil center found as the pupil boundary. 

6. The outer boundary is then drawn by adding 53 to the radius of the inner circle. 

7. The list “boundary” stores all the images with boundaries drawn, while the list “centers” stores the center coordinates.


## IrisNormalization.py
====================

We sequentially load each image from the list boundary returned by the previous function and initialize an empty list to store the normalized images. 

1. In order to project the polar coordinates onto the cartesian plane, we only need to focus on the region between the boundaries. 

2. We define an equally spaced interval over which the for loop iterates to convert polar coordinates to cartesian coordinates, using x=rcos(theta) and y=rsin(theta). 

3. We resize the image to a rectangular 64x512 sized image.


ImageEnhancement.py
===================

In this function, we enhance the image using Histogram Equalization to increase the contrast of the image for better feature extraction.


## FeatureExtraction.py
====================

The functions ‘m’ and ‘gabor’ help in calculating the spatial filter
defined in the paper

Code: 
(Images/code1.png)

The function spatial takes f, dx and dy as parameters which are defined
in the paper. It creates a 8x8 block which we run over the normalized
image to extract features.

As we have eyelashes and other occlusions present in normalized images,
we run this 8x8 block over a normalized image of 48\*512, which is our
region of interest. This helps in further improving the results of
matching.

Code: 
(Images/code2.png)

The function above defines this Region of interest and creates two
channels: filter1 and filter 2, which are then convolved with our RoI to
get two filtered images.

Code snippet for get\_vec(): 
(Images/code3.png)

These filtered images are then used to get our feature vector using the
function ‘get\_vec()’ This function calculates the mean and standard
deviation values grid by grid, where each grid is a 8x8 block. The
calculated values are then appended sequentially to the feature vector,
which is of size 1536 (6x64x4).


## IrisMatching.py
===============

This py file matches our testing and training feature vectors (with and
without dimensionality reduction).


The function dim\_reduction(feature\_vector\_train,feature\_vector\_test,components) does the following: 
It fits our LDA model to training data.
Then it transforms both training and testing feature vectors to the number of components specified in the parameters (max 107)

The function IrisMatching(feature\_vector\_train,feature\_vector\_test,components,flag) does the following: 

1. If flag==1 then it performs matching on the
original feature vector of size 1536. If flag==0 then it performs
matching on the reduced feature vector of size provided by the
components parameter. 

2. To perform matching, we calculate the L1, L2
and cosine distances as defined in the paper for every test image
against all training samples. Then, the minimum value for all 3
distances is taken as the matched image and its index is stored. 

3. Next, if the matched index for each image is correct (i.e, the matched
image is from the same folder as our test image) then it is considered
as the correct match, and 1 is appended to our “match” list. The same is
repeated for all three distances to get match\_L1,match\_L2 and
match\_cosine and they are returned 

4. We also calculate ROC matching here. If the cosine distance is less than the threshold, then it is 1
(accepted) otherwise it is 0 (rejected). This is stored for all 3 threshold values and returned.


## PerformanceEvaluation.py
========================

This py file calculates the correct recognition rates for our code. The
function PerformanceEvaluation(match\_L1,match\_L2,match\_cosine) does
the following:

1. The value of the correct correction rate would be
given by the count of the eyes that are correctly matched divided by the
count of the total number of eyes. This is calculated by dividing the
length of correct\_L1(as it has only those eyes that are correctly
matched) by length of match\_L1(as it has all the eyes). 

2. Thus, we get the values of crr\_L1,crr\_L2 and crr\_cosine.

## IrisRecognition.py
==================

This py file is the main file, where we call all the above functions to execute the entire function of iris recognition. 

### 1.
First, we read and process all training files:

a. Read all files 

b. Run iris localization on all of them and get the images.

c. On the localized images, run normalization and enhancement to get our enhanced images for feature extraction.

d. Then, run feature extraction to get all the feature vectors. 

### 2.
The same steps a-d are followed for the testing data. 

### 3.
Once we have our training and testing feature vectors, we run iris matching and performance evaluation on them as:

a. Get matching values and then CRR scores for 10,40,60,80,90 and 107 components in the reduced feature vector 
  
b. Use those values to plot the CRR vs Feature Vector dimensions graph 
  
c. Get matching values and then CRR scores for our full length 1536 feature vector 

d. Use the 1536 component CRR’s and 107 component CRR’s to plot the table to compare their accuracy 
  
### 4.
ROC requires the rate of false matches and the rate of false non-matches. 

False matches are the number of eyes that are matched but are not authorized whereas
False non-matches are the number of eyes that are rejected but are
authorized. To calculate ROC, we use the matching\_cosine\_ROC we got
from IrisMatching() and compare it with our actual matching\_cosine
answer to calculate the FMR and FNMR for all three threshold values.

FMR= no. of images incorrectly accepted / total number of accepted
images

FNMR = no. of images incorrectly rejected / total no of rejected images


## Images
======

This file contains 8 images depicting the step by step output we
obtained for localization, normalization and enhancement.

Fig1 depicts the grayscale image of the eye

Fig2 depicts the colored image which is what get stored in the target
array in the Localization function

Fig3 depicts the output of Localization, i.e the original image with
inner and outer boundaries

Fig4 depicts the Enhanced Normalized 64x512 rectangular image which is
used for further feature extraction steps

Fig5 shows the recognition results using features of different
dimensionality

Fig6 table that shows the car for L1,L2,cosine for original and reduced
feature vectors

Fig7 shows the ROC curve

Fig8 table that shows ROC fmr and fnmr measures for different
thresholds.
