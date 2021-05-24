
# Applying "Glitter Cartoon Filter" on an image.

Converting an image into a glitter cartoon filtered image using OpenCv, Os, Matplotlib and Numpy.

## Steps:
* Firstly imported the required libraries which are Numpy, Os, Matplotlib and Cv2.
* Taking path of the image/Real image as input using os and finally reading it using cv2

## Methods Used
* Used Bilateral Filter
* Followed by Median Blur
* Followed by Adaptive Threshold
* Followed by Bitwise "or" between original image and image_rgb
* And at last used Bitwise "and" between image_edge and output of the above "bitwise or image"
* Finally converted the image into "Glitter Cartoon Filtered" image




## Comparision between the "Original" and "Glitter Cartoon Filtered" Image
<img src="Images/Final_Output.jpg" height="500px">

