# Bunny Nose Snapchat Filter Using Computer Vision Techniques
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/Bunny-nose4.png" align="left" height="450px"/>

## Introduction 

Social media platforms such as Instagram and Snapchat are visual-based
social media platforms that are popular among young women. One popular
content many young women use on Snapchat and Instagram are beauty filters. A
beauty filter is a photo-editing tool that allows users to smooth out their skin,
enhance their lips and eyes, contour their nose, alter their jawline and
cheekbones, etc. Due to these beauty filters, young women are now seeking
plastic surgeons to alter their appearance to look just like their filtered photos
(this trend is called Snapchat dysmorphia)
 Overall, this study’s findings explain how beauty filters,
fitspirations, and social media likes affect many young women’s perceptions of
beauty and body image. By understanding why many young women use these
beauty filters it can help and encourage companies to create reliable resources
and campaigns that encourage natural beauty and self-love for women all around
the world. 




## History 



## How does Snapchat recognize a face?
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/Bunny.png" width="480px" align="right"/>

- his large matrix of numbers are codes, and each combination of the number represents a different color.
- The face detection algorithm goes through this code and looks for color patterns that would represent a face.
- Different parts of the face give away various details. For example, the bridge of the nose is lighter than its surroundings. The eye socket is darker than the forehead, and - - the center of the forehead is lighter than its sides.
- This could take a lot of time, but Snapchat created a statistical model of a face by manually pointing out different borders of the facial features. When you click your face on the screen, these already predefined points align themselves and look for areas of contrast to know precisely where your lips, jawline, eyes, eyebrows, etc. are. This statistical model looks something like this.
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Snapchat_Filters/Bunny_Nose/Images/face-landmark.png" align="right"/>
Once these points are located, the face is modified in any way that seems suitable.

## How to setup on Local Environment 


## Code Overview 

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

**Open Source First**

 best practices for managing all aspects of distributed services. Our shared commitment to the open-source spirit push the Face-X community and its projects forward.</p>
