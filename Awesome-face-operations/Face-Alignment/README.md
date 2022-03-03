# Face Alignment

## Problem Statement

1. Images given as input shall be centered.
2. Image snippets of faces shall be rotated and their eyes should lie on a horizontal line.
   
## Approach

To localize the faces in the image we are applying a pre-trained HOG + Linear SVM object detector specifically for the task of face detection. The user can even use any deep-learning based algorithms for face localisation.

Assuming we have completed the above process and received coordinates of the bounding box. We immeadiately move to the process of identifying the landmarks of the face. For the landmark detection we are going to use the official dlib 68 point facial landmark predictor which is trained upon [iBUG 300-W dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).

With the landmarks predicted we can identify left and right images. Find the angle that the eyes are making with the horizontal. Using the angle rotation matrix can be extracted with the help of OpenCV.


**This is a two step process.**
1. Detect the face in the image and localize them while identifying geometrical structure and properties of different landmarks.
2. Attempting the alignment of the face based on the geometrical properties extracted from the latter step.

### Modular Class inside facealigner has been implemented and a method inside the class has been implemented efficiently to do so
   
## Libraries Used
<img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white"></img>
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white"></img>


## Usage 
This is a basic example and can be found in the [main.py](main.py)

```python
detector = dlib.get_frontal_face_detector()
try:
	predictor = dlib.shape_predictor(args["shape_predictor"])
except:
	traceback.print_exc()
	sys.exit()
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
try:
	image = resize(image, width=800)
except:
	traceback.print_exc()
	sys.exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```


## Outputs
<img src="./outputs/output.gif">
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
