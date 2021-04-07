### Intro:
Template Matching is a method for searching and finding the location of a template image in a larger image. It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image. Several comparison methods are implemented in OpenCV.

- If input image is of size (WxH) and template image is of size (wxh), output image will have a size of (W-w+1, H-h+1). 
- Take it as the top-left corner of rectangle and take (w,h) as width and height of the rectangle. That rectangle is your region of template.

Suppose you are searching for an object which has multiple occurances, `cv2.minMaxLoc()` won’t give you all the locations. In that case, we will use thresholding. 

### Example:
![res_mario](https://user-images.githubusercontent.com/60208804/113759937-47e13580-9733-11eb-9c1c-c2acf373c8e6.jpg)
