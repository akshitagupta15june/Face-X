# Overview
#### Facial Recognition using Open-cv and sklearn KNearestNeighbour model.
# Dependencies
- `pip install numpy`
- `pip install opencv-python`
- `pip install sklearn`
# Quick Start
- Make a folder and then:

       git clone https://github.com/akshitagupta15june/Face-X.git
       cd Recognition using KNearestNeighbors

- To collect live data run below command
       
      python live_data_collection.py (to collect live data)

  It will ask to enter name of sample you are showing, after inputing show sample to webcam (Make sure there is sufficient light so that webcam recognises it).
  Then press `C` to capture images, and `Q` to exit screen.
  The data will be saved in `face_data.npy` file in same directory.
  
  ![Capture](./images/Capture2.jpg)



- For pre-processed data.
      
      python Pre_proccessed_data_collection.py
  
  Make sure the samples should be of 640x480 pixels and axis should match exactly. [Warning:The model should be trained with atleast 5 images.]

- After data collection, run below command to train model and recognising images using webcam.
    
      python image_recogniser.py
      
 # Screenshot
![Capture](./images/Capture1.jpg)
![Capture](./images/Result.jpg)
