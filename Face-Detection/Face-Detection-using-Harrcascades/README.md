Hey everyone!

In this section, let us first discuss about face detection vs face recognition.


![Face detection vs Face recognition](https://images.squarespace-cdn.com/content/v1/56fbbea286db438eca867c34/1560417300787-OR1KD9FLLRFR3KTKF7QS/FaceDetect-vs-FacialRecog.jpg)

This image clearly represents the difference between face detection and face recognition.
Still, let me clear the difference.

>**Face Detection:**
This detects the precense of a face, it returns either a true or false value.

>**Face recognition:**
This is a more advance version of face detection which in addition to detection of face, also identifies the person based on the previous data used to train the model.

<img align="center" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="javascript" width="100%"/>

**STEPS TO RUN THE SCRIPT:**

**Step 1:** Make sure that opencv is installed in your system. If it is not, then refer to this [click here.](https://www.geeksforgeeks.org/how-to-install-opencv-for-python-in-windows/)

**Step 2:** Fork this repo by clicking FORK button on the top right.

**Step 3:** Go to the forked repository (Make sure it is showing as <Your-GitHub-Username>/Face-X). Click on the green button with code label and copy the link.

**Step 4:** Open git bash on your system and clone the repo using ```git clone <paste-url>``` ([click here](https://www.educative.io/answers/how-to-install-git-bash-in-windows) to learn about how to install git bash).

**Step 5:** Go to the Face-X directory using ```cd Face-X/Face-Detection/Face-Detection-using-Harrcascades/```.

**Step 6:** Run the face_detect.py script using ```python3 face_detect.py``` command ([Download and install python from here](https://www.python.org/downloads/)).

**Step 7:** Press any key to exit.

<img align="center" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="javascript" width="100%"/>

STEPS TO ADD YOUR OWN IMAGE AND PLAYING WITH THE CODE:

*To change the input image to your own image/image of your choice:*

**Step 1:** Copy the image you want to use and renameit to something simple like ```test_image.jpg```.

**Step 2:** While on same directory, use ```nano face_detect.py``` command in git bash.

**Step 3:** Nano text editor will open. Scroll down to 9th line (use arrow keys on your keyboard) and change the name of ```cast2.jpg``` to ```test_image.jpg```.

**Step 4:** To exit nano text exitor, press ```CTRL+X``` then press ```y``` and then press ```enter```. Now the modified file will be saved in the directory.

**Step 5:** Re-run the face_detect.py file using the **Step 6** from above portion.



**NOTE:**

At this point, you might have noticed that it detects many false faces in the images or might have skipped some faces. This is due to the limitation of computer vision. It can be fixed upto an level.


To make these fixes, you'll need to change the ```faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors = 6)``` line from the code. Open the file using nano (refer to Step 2 from above protion).

If you want it to detect more faces, change the value of minNeighbors parameter to lower number than 6. Or the best way to explore is to play with it yourself ;-)

Just remember to save the file before running the script again. (Remember ```CTRL+X```>```y```>```Enter```).




I would like to share some of my own results:

![Screenshot 1](https://user-images.githubusercontent.com/74321084/193473953-e21d3e67-619d-4f26-b31e-4638ea66edac.PNG)

```cast.jpg``` with ```minNeighbors=6```

  
  
![Screenshot 2](https://user-images.githubusercontent.com/74321084/193473995-ebb23944-89c0-4513-b13f-6c2dd7b38a12.PNG)
  
```lady.jpg``` with ```minNeighbors=6```
  

![Screenshot 3](https://user-images.githubusercontent.com/74321084/193474107-f666935a-8c13-43ba-a8c1-68c23d725fbd.PNG)
```lady.jpg``` with ```minNeighbors=26```
  
  
![Screenshot 4](https://user-images.githubusercontent.com/74321084/193474128-7cd3ce16-869e-4da3-97e0-288beba540d2.PNG)
```cast2.jpg``` with ```minNeighbors=6```
  
  
  
  
  
  
Feel Free to contribute and play with the project.
