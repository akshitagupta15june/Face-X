# Bunny Nose Snapchat Filter Using Computer Vision Techniques
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/Bunny-nose4.png" align="left" height="450px"/>

## 1.Introduction 

Social media platforms such as Instagram and Snapchat are visual-based
social media platforms that are popular among young women. One popular
content many young women use on Snapchat and Instagram are beauty filters. A
beauty filter is a photo-editing tool that allows users to smooth out their skin,
enhance their lips and eyes, contour their nose, alter their jawline and
cheekbones, etc. Due to these beauty filters, young women are now seeking
plastic surgeons to alter their appearance to look just like their filtered photos
(this trend is called `Snapchat dysmorphia`)
 Overall, this study’s findings explain how beauty filters,
fitspirations, and social media likes affect many young women’s perceptions of
beauty and body image. By understanding why many young women use these
beauty filters it can help and encourage companies to create reliable resources
and campaigns that encourage natural beauty and self-love for women all around
the world. 
<br><br/>
## 2.History 

`Face filters` are augmented reality effects enabled with face detection technology that overlay virtual objects on the face. Introduced by Snapchat back in 2015 as a fun way to dress-up your selfie, today face filters have become a meaningful tool to improve digital communication and interaction. 
       Many people use social media apps such as Instagram or Snapchat, which have face filters for people to take and post pictures of themselves. But many people do not realize how these filters are created and the technology behind how they fit people’s faces almost perfectly. The mechanics behind face filters was originally created by a Ukrainian company called Looksery; they used the technology to photoshop faces during video chats. Snapchat bought their algorithm, called the `Viola-Jones algorithm`, and created the face filters seen in many social media apps today.

Creating face filters is more difficult than you may think, so I’ll break it down into five key steps:

- The first step is face detection. The image is initially viewed in ones and zeros, so the algorithm scans the image, looking specifically for color patterns. This can include finding that the cheek is lighter than the eye or that the nose bridge is lighter than surrounding areas. After detecting these patterns, a face can be distinguished in the camera.
- The second step is the `landmark extractio`n. Using specific algorithms in a 2D image, facial features such as the chin, nose, forehead, etc are determined.
- The third step is face alignment. The coordinates of landmarks on people’s faces are taken to properly fit the filter to a particular face.
- The fourth step is 3D mesh. Using the 2D image, a 3D model of the user’s face is built to fit the filter animation to a specific face.
- The last step is face tracking, which approximates and locates the 3D mask in real time. This allows the user to move their face without the filter disappearing or moving to an incorrect location.

Another way to think of these steps is to imagine a human body. The landmarks identified in a 2D image serve as the skeleton for the future mask. Similar to how bodies differ in shape, so do people’s face structures. Using face alignment, the filter matches with the coordinates of landmarks from a certain face. People’s skin makes them look the way they are and 3D mesh step is like aligning the skin to the skeleton. Similar to how bodies move while keeping the skeleton, skin and muscle together, face tracking follows the face to make sure the filter stays on the right coordinates.


## 3.How does Snapchat recognize a face?
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/Bunny.png" width="480px" align="right"/>

- his large `matrix` of numbers are codes, and each combination of the number represents a different color.
- The face detection algorithm goes through this code and looks for color patterns that would represent a face.
- Different parts of the face give away various details. For example, the bridge of the nose is lighter than its surroundings. The eye socket is darker than the forehead, and - - the center of the forehead is lighter than its sides.
- This could take a lot of time, but Snapchat created a statistical model of a face by manually pointing out different borders of the facial features. When you click your face on the screen, these already predefined points align themselves and look for areas of contrast to know precisely where your lips, jawline, eyes, eyebrows, etc. are. This `statistical model` looks something like this.
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/face-landmark.png" height="400px" align="right"/>
Once these points are located, the face is modified in any way that seems suitable.

## 4.How to setup on Local Environment :

- Fork and Clone the repository using 
```
git clone https://github.com/akshitagupta15june/Face-X.git
```
- Create virtual environment 
```
- python -m venv env
- source env/bin/activate (Linux)
- env\Scripts\activate (Windows)

```
- Go to project directory
```
cd  Sanpchat_Filters/Bunny_Nose 

```
- Run Program
```
py Bunny_Nose_Filter.py
```

### Step 1: Find Faces in a Picture
Now that we know the basics of how computer vision works, we can begin to build our filter. First, let’s find faces and eyes in a static picture. Begin by installing on your computer and then importing OpenCV (an open-source python package for image processing) into a py file. All image structures in OpenCV are can be converted to and from NumPy arrays so it may be useful to import NumPy as well. Once you’ve installed OpenCV, you should have access to .xml files that contain facial recognition and other image processing algorithms. For this tutorial, we’ll use an algorithm called the Haar Cascade for faces and eyes. If you are having trouble finding the directory where these .xml files are, I suggest a quick file search for` “haarcascade”`. Once you have found the path to the directory where your Haar Cascades are stored, call CascadeClassifier and pass the path through it:

```
import cv2
import numpy as np 

#path to classifiers
path = '/Users/mitchellkrieger/opt/anaconda3/envs/learn-env/share/opencv4/haarcascades/'

#get image classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')
```
Great, now that we’re set up, we can load in the images and look for faces. Note that Haar Cascades and many facial recognition algorithms require images to be in grayscale. So, after loading in the image, convert it to `grayscale`, and then use the face_cascade to detect faces. After you’ve got the faces, draw a rectangle around it and search within the facial region for eyes. Then draw rectangles around each eye.

The Tools of Face Detection in Python: 

  We’ll use two of the biggest, most exciting image processing libraries available for` Python 3`, `Dlib` and `OpenCV`.

Installing Dlib is easy enough, thanks to wheels being available for most platforms. Just a simple pip install dlib should be enough to get you up and running.

For OpenCV, however, installation is a bit more complicated. If you’re running on MacOS, you can try this post to get OpenCV setup. Otherwise, you’ll need to figure out installation on your own platform.

we are going to use dlib and OpenCV to detect facial landmarks in an image.

Facial landmarks are used to localize and represent salient regions of the face, such as:

- `1.Eyes`
- `2.Eyebrows`
- `3.Nose`
- `4.Mouth`
- `5.Jawline`
- `6.Facial landmarks have been successfully applied to face alignment, head pose estimation, face swapping, blink detection and much more.`






### Step 2: Create your image filter
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/eyeflow.png" height="450px" align="right"/>

We’ll use OpenCV to get a raw video stream from the webcam. We’ll then resize this raw stream, using the imutils resize function, so we get a decent frame rate for face detection.

Once we’ve got a decent frame rate, we’ll convert our webcam image frame to black and white, then pass it to Dlib for face detection.

Dlib’s get_frontal_face_detector returns a set of bounding rectangles for each detected face an image. With this, we can then use a model (in this case, the` shape_predictor_68_face_landmarks on Github`), and get back a set of 68 points with our face’s orientation.

From the points that match the eyes, we can create a polygon matching their shape in a new channel.

With this, we can do a bitwise_and, and copy just our eyes from the frame.

We then create an object to track the n positions our eyes have been. OpenCV’s boundingRect function gives us a base x and y coordinate to draw from.

Finally, create a mask to build up all the previous places where our eyes where, and then once more, bitwise_and copy our previous eye image into the frame before showing.

### Step 3: Put the model into action

Finally, we’ll look at some results of applying facial landmark detection to images.

What are facial landmarks?

Figure 1: Facial landmarks are used to label and identify key facial attributes in an image (source).
Detecting facial landmarks is a subset of the shape prediction problem. Given an input image (and normally an ROI that specifies the object of interest), a shape predictor attempts to localize key points of interest along the shape.

In the context of facial landmarks, our goal is detect important facial structures on the face using shape prediction methods.

`Detecting facial landmark`s is therefore a two step process:

- `Step 1`: Localize the face in the image.
- `Step 2`: Detect the key facial structures on the face ROI.
- `Face detection` (Step #1) can be achieved in a number of ways.

We could use OpenCV’s built-in Haar cascades.

We might apply a pre-trained HOG + Linear SVM object detector specifically for the task of face detection.

Or we might even use deep learning-based algorithms for face localization.

In either case, the actual algorithm used to detect the face in the image doesn’t matter. Instead, what’s important is that through some method we obtain the face bounding box (i.e., the `(x, y)-coordinates` of the face in the image).

Given the face region we can then apply Step #2: detecting key facial structures in the face region.

There are a variety of facial landmark detectors, but all methods essentially try to localize and label the following facial regions:
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/Bunny-nose3.png" height="400px" align="right"/>

- `Mouth`
- `Right eyebrow`
- `Left eyebrow`
- `Right eye`
- `Left eye`
- `Nose`
- `Jaw`

We got our model working, so all we gotta do now is use OpenCV to do the following:
- Get image frames from the webcam
- Detect region of the face in each image frame because the other sections of the image are useless to the model (I used the Frontal Face Haar Cascade to crop out the region of the face)
- Preprocess this cropped region by — converting to grayscale, normalizing, and reshaping
- Pass the preprocessed image as input to the model
- Get predictions for the key points and use them to position different filters on the face

This method starts by using:

- A training set of labeled facial landmarks on an image. These images are manually labeled, specifying specific (x, y)-coordinates of regions surrounding each facial structure.
- Priors, of more specifically, the `probabilit`y on distance between pairs of input pixels.

Given this training data, an ensemble of regression trees are trained to estimate the facial landmark positions directly from the pixel intensities themselves (i.e., no “feature extraction” is taking place).

The end result is a facial landmark detector that can be used to detect facial landmarks in real-time with high quality predictions.

## 5.Code Overview 

```
import cv2
import numpy as np
import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
nose_image = cv2.imread("bunny.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # Nose coordinates
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)

        nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1]) * 1.7)
        nose_height = int(nose_width * 0.77)

        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2),
                              int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2),
                       int(center_nose[1] + nose_height / 2))


        # Adding the new nose
        nose_bunny = cv2.resize(nose_image, (nose_width, nose_height))
        nose_bunny_gray = cv2.cvtColor(nose_bunny, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_bunny_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_bunny)

        frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose

        cv2.imshow("Nose area", nose_area)
        cv2.imshow("Nose bunny", nose_bunny)
        cv2.imshow("final nose", final_nose)



    cv2.imshow("Frame", frame)



    key = cv2.waitKey(1)
    if key == 27:
        break
```


## Result Obtain






































<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/Bunny-nose5.png" height="400px" align="left"/>
<p style="clear:both;">
<h1><a name="contributing"></a><a name="community"></a> <a href="https://github.com/akshitagupta15june/Face-X">Community</a> and <a href="https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md">Contributing</a></h1>
<p>Please do! Contributions, updates, <a href="https://github.com/akshitagupta15june/Face-X/issues"></a> and <a href=" ">pull requests</a> are welcome. This project is community-built and welcomes collaboration. Contributors are expected to adhere to the <a href="https://gssoc.girlscript.tech/">GOSSC Code of Conduct</a>.
</p>
<p>
Jump into our <a href="https://discord.com/invite/Jmc97prqjb">Discord</a>! Our projects are community-built and welcome collaboration. 👍Be sure to see the <a href="https://github.com/akshitagupta15june/Face-X/blob/master/Readme.md">Face-X Community Welcome Guide</a> for a tour of resources available to you.
</p>
<p>
<i>Not sure where to start?</i> Grab an open issue with the <a href="https://github.com/akshitagupta15june/Face-X/issues">help-wanted label</a>
</p>
