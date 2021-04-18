# Face-Recognition-using-FaceNet


This face recognition system is implemented upon a pre-trained FaceNet model achieving a state-of-the-art accuracy.
The system comes with
both Live recognition & Image recognition.
It is trained on faces of some celebrities.

For any queries Contact: [Ankur Goswami](https://github.com/Ankur1401/)


* __Installing dependencies:__
  * For Anaconda users: `conda install --file requirements.txt`<br>
  * For python users: `pip install -r requirements.txt`<br>
    (even Anaconda users can use this if they use anaconda prompt instead of terminal)

* __Downloading the model__:<br>
  The repository requires an additional file to work. The file is too large to upload here.
  So I've provided a Google Drive link of it. Download the file and keep it inside [`/data/model/`](https://github.com/Ankur1401/Face-Recognition-using-FaceNet/tree/master/data/model) directory.<br> [Click Here](https://drive.google.com/open?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1)  to download the file.
  
* __Training on other faces:__ <br>
To train model on different faces, follow the given steps:<br>
  1. Put the images containing clear frontal face in [`/data/images/`](https://github.com/Ankur1401/Face-Recognition-using-FaceNet/tree/master/data/images) directory.
  1. Open the repository directory in terminal and run following commands in given order:
     1. `cd script`
     1. `python generate_data.py`
  1. Follow program instructions.
  
* __Testing/Detecting faces:__ <br>
  1. __Face Recognition from Images__:
     1. Put the images containing the faces to predict in [`/test/`](https://github.com/Ankur1401/Face-Recognition-using-FaceNet/tree/master/test) directory.
     1. Open the repository directory in terminal and run following command:
      ```
          python image_recognition.py
      ```
     1. Output images will then be available in [`/test/predicted/`](https://github.com/Ankur1401/Face-Recognition-using-FaceNet/tree/master/test/predicted) directory.
   
  1. __Live Face Recognition(Obviously using camera):__
   <br>Open the repository directory in terminal and run following command:
      ```
      python live_recognition.py
      ```

## Examples:

__NOTE:__ Faces with __Unidentified__ labels are faces on which the model is not trained.

__Example #1:__
<br>Before:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/vampire-diaries.jpg width=50%>
<br>After:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/predicted/vampire-diaries.jpg width=50%>

__Example #2:__
<br>Before:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/the-avengers-walt03.jpg width=50%>
<br>After:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/predicted/the-avengers-walt03.jpg width=50%>

__Example #3:__
<br>Before:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/2ec945ecd5f7c08789f3ef5da5287410.jpg width=50%>
<br>After:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/predicted/2ec945ecd5f7c08789f3ef5da5287410.jpg width=50%>

__Example #4:__
<br>Before:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/3840x2160-avengers-cast-photocall-moscow-05.JPG width=75%>
<br>After:(Need to zoom)<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/predicted/3840x2160-avengers-cast-photocall-moscow-05.JPG width=75%>

__Example #5:__
<br>In this example, the model was trained on faces of my friends.
<br>Before:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/00000PORTRAIT_00000_BURST20180310144236486.jpg width=75%>
<br>After:<br>
<img src=https://github.com/Ankur1401/Face-Recognition-using-FaceNet/blob/master/test/predicted/00000PORTRAIT_00000_BURST20180310144236486.jpg width=75%>

