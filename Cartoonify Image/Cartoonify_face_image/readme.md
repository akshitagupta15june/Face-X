# Cartoonifying a face image.

- To Cartoonify the image we have used Computer vision, Where we apply various filters to the input image to produce a Cartoonified image. To accomplish this we have used Python programming language and opencv a Computer Vision library.<br>
 
 ## Dependencies:
 The Dependencies used are:
 - Opencv :It provides the tool for applying computer vison techniques on the image.
 - Numpy :Images are stored and processed as numbers, These are taken as arrays.

## How to Run:
- Download the directory.
- You can use any Editor, Notebook Or IDE's to open the Cartoonify-face_image.py file.
- Run Cartoonify-face_image.py file.
- Press Space bar to exit.

## Steps of its working:

- We have imported the cv2 and numpy library.
- We are capturing the image frames using cv2.VideoCapture().
- We are reading the image frames by using frame_cap.read().
- We are applying Gray scale filter to the image frames using cv2.cvtcolor() and the by passing second parameter as cv2.COLOR_BGR2GRAY.
- We are using MedianBlur on the gray scale image obtained above by setting the kernal size as 5 to blur the image using cv2.medianBlur().
- We are using adaptive threshold on the image obtained after applying Medianblur, we are using a threshold value of 255 to filter out the pixel and we are using the adaptive method cv2.ADAPTIVE_THRESH_MEAN_C with a threshold type as cv2.THRESH_BINARY  and block size 9.
- We are applying a Bilateral filter on the original image frames using cv2.bilateralFilter() with kernal size 9 and the threshold as 250 to remove the Noise in the image.
- We are then applying Bitwise and operation on the Bilateral image and the image obtained after using Adaptive threshold which gives the resulting cartoonified image.

## Result:
 ### Input Video
 ![Actual Video](https://user-images.githubusercontent.com/62636670/111106310-e1755700-857a-11eb-8ac7-3452d3430592.gif)

 ### Output Video
 ![Cartoonified Video](https://user-images.githubusercontent.com/62636670/111105513-26988980-8579-11eb-849a-c4babf925260.gif)
 
 
  
