## Overview
    Face detection and recognition using LBP features and Support Vector Machine.
    This model first creates the rectangle and crop the facial area from image then extracts LBP features from image and pass it through SVM.

## Dependencies
    pip install numpy
    pip install opencv-python
    pip install sklearn
    pip install skimage
    
    
## Quick Start
    1] git clone https://github.com/akshitagupta15june/Face-X.git
    2] cd Recognition using LBP_SVM
    
    
    -->To collect live data run below command
    3] python create_dataset.py
      (First it will ask your name and then after it will take your 100 snapshots. You can press 'Esc' to exit)
    
    -->After data collection, run below command to train model and recognising images using webcam.
    4] python model.py
    
   
