
# Snapchat Filters using OpenCV

### Dependencies

* Python
* The program makes use of Dlib-facial feature points
* OpenCV



### Installing
* Git clone repository
```
git clone 
```
* Make sure to install the dependencies:
```
pip install dlib
```
* Any modifications needed to be made to files/folders
```
pip install opencv-python
```



To run :

- `facial_test.py`          <--- inital script that finds faces and eyes in images
- `halloween_masks.py`      <--- edited `facial_test.py` script that adds `witch.png` to faces
- `halloween_masks_vid.py`  <--- edited `halloween_masks.py` script with live video feed instead of static images
- `saved.png`               <--- static image of Saved by the Bell Cast used for testing
- `witch.png`               <--- filter image to be placed on faces
- `witch.gif`               <--- gif of screen recording of `halloween_masks_vid.py`
