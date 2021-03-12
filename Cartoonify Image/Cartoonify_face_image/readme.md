# Cartoonifying a face image.

- To Cartoonify the image we have used Computer vision, Where we apply various filters to the input image to produce a Cartoonified image. To accomplish this we have used Python programming language and opencv a Computer Vision library.<br>
 
 ## Dependencies:
 The Dependencies used are:
 - Opencv :It provides the tool for applying computer vison techniques on the image.
 - Numpy :Images are stored and processed as numbers, These are taken as arrays.

## How to Run:
- Download the directory.
- You can use any Editor, Notebook Or IDE's to open the `Cartoonify-face_image.py` file.
- Open the `Cartoonify-face_image.py` file and in the cv2.imread() put the path of your image which you want to Cartoonify.
- Run `Cartoonify-face_image.py` file.

## Steps of its working:

- We have imported the cv2 and numpy library.
- We are reading the image using `cv2.imread()`.
- We are applying Gray scale filter to the image using 'cv2.cvtcolor()' and the by passing second parameter as `cv2.COLOR_BGR2GRAY`.
- We are using MedianBlur on the gray scale image obtained above by setting the kernal size as 5 to blur the image using `cv2.medianBlur()`.
- We are using adaptive threshold on the image obtained after applying Medianblur, we are using a threshold value of 255 to filter out the pixel and we are using the adaptive method cv2.ADAPTIVE_THRESH_MEAN_C with a threshold type as cv2.THRESH_BINARY  and block size 9.
- We are applying a Bilateral filter on the original image using `cv2.bilateralFilter()` with kernal size 9 and the threshold as 250 to remove the Noise in the image.
- We are then applying Bitwise and operation on the Bilateral image and the image obtained after using Adaptive threshold which gives the resulting cartoonified image.

## Result:
 ### Input Image
 <img src="https://github.com/akshitagupta15june/Face-X/blob/master/Cartoonify%20Image/Cartoonify_face_image/Images/face.jpg" width="500" height="300" />
  
 ### Output Image
  ![after_face](https://user-images.githubusercontent.com/62636670/110942958-93850700-8360-11eb-935b-5e8ca089dc01.png)
 
