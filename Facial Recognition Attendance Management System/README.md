
## About:🤔💭
- Facial recognition attendance management system. It makes use of openCV and python to recognise face. 

- User has to:

  - Click on 'Automatic attendance'.
  - Enter the subject name.
  - Click on 'Fill attendance' and wait for camera window to open.
  - It will recognise the face and display enrollment number and name.
  - Click on 'Fill attendence'.

----

### List TO-DO📄:

- [x] Check whether your system has a web cam or not. If not, then get one camera for face recognisation.
- [x] Install [Python.](https://www.howtogeek.com/197947/how-to-install-python-on-windows/)
- [x] Install [Dependencies.](https://github.com/smriti1313/Face-X/blob/master/Facial%20Recognition%20Attendance%20Management%20System/README.md#dependencies)
- [x] [Download](https://www.wikihow.com/Download-a-GitHub-Folder) [Face-X](https://github.com/akshitagupta15june/Face-X) and open `Facial Recognition Attendance Management System folder`.
- [x] Create a **TrainingImage** folder in this folder.
- [x] Open a **run.py** and change the all paths with your system path.
- [x]  Run run.py.

----

### Requirements:🧱🧱

|Hardware|Software|
|----|-----|
|web cam or camera|python|

----

### Dependencies🔧🛠:
Open terminal and write:

* `pip install Pillow`
* `pip install opencv-python`
* `pip install pandas`
* `pip install pymysql`
* `pip install opencv-contrib-python`
* Tkinter already comes with python when python is downloaded.

----

## Testing🧰:

- After running, you need to give your face data to system so `enter your ID` and `name` in box. 
- Then click on `Take Images` button.
- It will collect 200 images of your faces and will save it in `TrainingImage folder`.
- After that we need to train a model (to train a model click on `Train Image` button.)
- It will take 5-10 minutes for training(for 10 person data).
- After training click on `Automatic Attendance` ,it can fill attendace by your face using our trained model (model will save in `TrainingImageLabel`). It will create .csv file of attendance according to time & subject.
- You can store data in database (install wampserver),change the DB name according to your in `run.py`.
- `Manually Fill Attendace Button` in UI is for fill a manually attendance (without face recognition),it's also create a .csv and store in a database.


>For better understanding watch [this.](https://www.youtube.com/watch?v=dXViSRRydRs)
