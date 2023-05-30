
# Speaker Detection

Speaker Detection is a script that identifies the people speaking in a frame by turning their lips from the color green to red. It is mainly based on the library [cv2](https://opencv.org/) and uses [dlib](http://dlib.net/) as it's main facial detector.

## Objectives

The main objectives of this project are:

- To detect person who is speaking from many among the frame
- To gain knowledge on the usage of opencv library
- To use dlib library as a facial features detector

## Limitations

- Script must be run in a well-lit environment
- Appropriate distance must be maintained from the camera
- Excessive movement should be avoided

## Usage

The script uses gym module. To install please type the following command in your bash.

```bash
pip install requirements.txt
```

Download ``shape_predictor_68_face_landmarks.dat`` from any github directory

After which, once you have navigated to the directory, you can simply run the script using 

```bash
python3 main.py
```
To exit from the program, press key ``q`` while webcam is open

## Author

Contributed by [Mehlam Songerwala](https://github.com/mehlams/)