
# README

* Libraries used:
    * OpenCV

* Created a Happy Diwali filter using OpenCV

* Requirements:
    * opencv_python==4.6.0.66
    
* Steps of Implimentation
   * Getting video feed through webcam using :
      * `VideoCapture()`
   * Getting the dimensions of the feed through the webcam using :
      * `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)`
   * Selecting all the 3 color channels in the frame & mask :
      * `frame_mask = img[:, :, 2]` `mask_img = img[:, :, 0:3]`
   * Redefining the dimensions :
      * `wd = int(width)` `ht = int(height)`
   * Using the redefined dimensions for frame & mask :
      * `diwali = cv2.resize(mask_img, (wd, ht))` `mask = cv2.resize(frame_mask, (wd, ht))`
   * Obtaining the region-of-interest.
   * Adding filter to the frame.
      * `frame[y1:y2, x1:x2] = cv2.add(frame1, frame2)`
   * Showing the Output Window using :
      * `cv2.imshow('Snapchat Filter Diwali', frame)`
   

SCREENSHOT OF FILTER :

![hampy_diwali](https://user-images.githubusercontent.com/105866331/213911339-94d2a4f3-4286-42d5-83b9-34aa211d3d38.jpg)

SCREENSHOT OF USING THE FILTER :

<img width="549" alt="Happy_Diwali_Filter" src="https://user-images.githubusercontent.com/105866331/213911344-e46084a0-8ec1-4e9b-890f-7ef82879c37f.png">
