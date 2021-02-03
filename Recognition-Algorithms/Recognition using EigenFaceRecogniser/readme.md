Face Recognition using EigenFaceRecognizer

## Requirements

    Python3.6+
    virtualenv (pip install virtualenv)

## ScreenShots

![](https://i.imgur.com/D2iECmo.png)
![](https://i.imgur.com/L8UVy93.png)

## Installation

    virtualenvv env
    source venv/bin/activate (Linux)
    venv\Scripts\activate (Windows)
    pip install numpy==1.19.2
    pip install opencv-contrib-python==4.4.0.44
    pip install opencv-python==4.4.0.44
    pip install python-dotenv==0.14.0
    
## Execution

Update the datapath accordingly as in the files used in this repository ,follow the data path for akshita as the main user.

In Ubuntu,
Meaning if there are data paths like:
        file_name_path='/home/akshita/Desktop/Face_reco/user'+str(count)+'.jpg'

Then Update it to:
        file_name_path='/home/yourusername/Desktop/Face_reco/user'+str(count)+'.jpg'

    python facial_recognition_part1.py (face images collection)
    python facial_recognition_part2.py (training)
    python facial_recognition_part3.py (final recognition)
