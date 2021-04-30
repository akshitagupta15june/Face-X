# Live-Face-Verification-Using-Deep-Learning

Face Detection and landmark detection : It is done using Multi-task Cascaded Convolutional Networks(MTCNN) model. Used a pretrained model of MTCNN to detect face, to find the bounding box and landmark detection.
Reference : https://arxiv.org/pdf/1604.02878.pdf
            https://github.com/ipazc/mtcnn

Face Verification : The face Recognition is done using Facenet model. Used a pretrained facenet model to compare the captured image with images in database to verify person's face
Reference : https://arxiv.org/pdf/1503.03832.pdf

1. MTCNN Pretrained Model : Follow link -https://github.com/ipazc/mtcnn
2. FaceNet Pretrained Model : https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk

Dependency :
 1. Python 3.6
 2. tensorflow r1.12 or above 
 3. OpenCV 4.1.1 or above

How to Run the Code :
 1. Download and extract the Face_verification.zip code into a folder.
 2. Download the pretrained model from the link given and place the files in the extracted folder in step 1.
 3. Input some images of person inside the database folder.
 4. Now execute the Live_Face_Verification.py script.

Execution:  
 1. The image captured can be seen inside captured folder.
 2. The detected face with bounding box, cropped part of image and land mark detected images can be seen inside folder check.

Note : 
 1. Use the port number for the webcam according to Device Configration of your sytem.(Edit :Face_Rec.py,  Line 9 ,                 camera=cv2.VideoCapture(0)). 
 2. Use webcam with resolution greater than 640x480 for better accuracy.
         
