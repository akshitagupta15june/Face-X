# Face-detection-using-vgg16
face detection using tensorflow

## files:  

1) face_detection.ipynb : in this file i trained whole model and validate with 99% accuracy.
2) live_face_detection.py : in this file i used opencv to live detection of the faces.
3) demo : demo image for testing.
4) screen_shot.png 1 & 2 : screen shot of live detection.
5) hasrcasecade_face_frontage_default.xml : this file used to detect face shape on live screen.

## Dependencies:
below modules you needs install to run my project.
1) tensorflow(1.13.1)
2) keras(2.3.1)
3) opencv2(4.1.1)
4) python(3.7.7) 
5) plateform Anaconda(4.7.11) 

## Steps:
please follow below step to make detection.
1) Download whole Repository
2) Download this dataset : https://drive.google.com/file/d/1MNjGcS4ffaHLYoVDXru-w_feE0z1qJba/view
3) Unzip the Dataset in this Repository
4) Open face_detection.ipynb on jupyter notebook and Run all cell one by one. 
   - Steps that I follow to make this notebooks
    1) imports libraries
    2) define variables(like image_size, train path, test path, etcs)
    3) Download vgg16
    4) make model
    5) read all images
    6) train the model with validation
    7) plot accuracy of training and validation sets
    8) test one pic to check accuracy (here i used only one photo because i have limited photos of me)
    9) save the model as 'Final_Model_Face.h5' (this file will be used in live detection)
    Note: if you got any error then check your path or module that you installed.
5) open live_face_detection.py on spyder.
    - Code explanation:
    1) importing libraries.
    2) read the model 'Final_Model_Face.h5'
    3) Make sure your webcam is working.
    4) read the haarcasecade file 'haarcascade_frontalface_default.xml'.(this file used to find human face on live video.)
    5) then this human face convert to array form and predicting using this 'Final_Model_Face.h5' model.
    6) whatever output(prediction) i get Ash, Malav or Nani, print on live cam.
    7) when you run this file it will start you webcam and on screen you can see the output.
    

Thank you.  
Ashish  
linkedin : https://www.linkedin.com/in/ashishbarvaliya/  
resume :https://drive.google.com/drive/u/0/folders/1UHBqfC4jcmNYe7L8YK0_Y9kpuXLBkNta  
