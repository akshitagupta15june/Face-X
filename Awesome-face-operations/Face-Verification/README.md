# Live Face Verification Using Deep Learning

Face Detection and landmark detection : It is done using Multi-task Cascaded Convolutional Networks(MTCNN) model. Used a pretrained model of MTCNN to detect face, to find the bounding box and landmark detection.
Reference :
* https://arxiv.org/pdf/1604.02878.pdf
* https://github.com/ipazc/mtcnn
* https://arxiv.org/pdf/1503.03832.pdf


Face Verification : The face Recognition is done using Facenet model. Used a pretrained facenet model to compare the captured image with images in database to verify person's face


1. MTCNN Pretrained Model : Follow link -https://github.com/ipazc/mtcnn
2. FaceNet Pretrained Model : https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk

## Dependency :
 * Python 3.6
 * tensorflow r1.12 or above 
 * OpenCV 4.1.1 or above

## How to Run the Code :
 1. Download and extract the Face_verification.zip code into a folder.
 2. Download the pretrained model from the link given and place the files in the extracted folder in step 1.
 3. Input some images of person inside the database folder.
 4. Now execute the Live_Face_Verification.py script.

## Execution:  
 1. The image captured can be seen inside captured folder.
 2. The detected face with bounding box, cropped part of image and land mark detected images can be seen inside folder check.


## Output: 

![Output](https://user-images.githubusercontent.com/65017645/116647486-4e607700-a998-11eb-8d8c-6cff772bb356.jpeg)
