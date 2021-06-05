# Face Recognition Using the Eigen Face Method
 We have designed a face recognition system using the eigenface method. For more information on eigen faece method refer this link :
 https://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf. We have a set of M training images and another set of test images. We used training images to produce a set of eigenfaces. Then we recognized the face in the input image using the eigenface method. We have used Euclidean distance as distance measure for computing 𝑑𝑖, for 𝑖 = 0 to 𝑀. You can manually choose the thresholds 𝑇0 and 𝑇1 that produce the best results.

## Getting Started
Clone the repo. One can simply download the FaceRecognition.py script and the test images. 


### Prerequisites

Python 3.

### Instructions for running

Run the following command:
python FaceRecognition.py 

This script will create four folders which contains following output images:
* Mean Face
* Eigen Face
* Reconstructed Test Faces 
* Mean Subtracted Test Faces 

One can maually set the two thresholds in the code for:
* ckassification of a  image as face/non-face image
* identity of a face

### Miscellaneous

A detailed report called "Face Recognition Project Report" is also provided. It lists all the steps to get the project running along wih a compilation of results for each test face. Have a look to get a better insight.

P.S - A blog post with a clear explananation of the Eigen Face method and a step by step code walkthrough is coming soon.