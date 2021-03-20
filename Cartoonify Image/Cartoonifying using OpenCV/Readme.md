Currently there are lots of professional cartoonizer applications available in the market but most of the them are not freeware. In order to get the basic cartoon effect, we just need the bilateral filter and some edge dectection mechanism. The bilateral filter will reduce the color palette, which is essential for the cartoon look and edge detection is to produce bold silhouettes.

We are going to use openCV python library to convert an RGB color image to a cartoon image.

Algorithm

Firstly apply the bilateral filter to reduce the color palette of the image.
Then conver the actual image to grayscale.
Now apply the median blur to reduce image noise in the grayscale image.
Create an edge mask from the grayscale image using adaptive thresholding.
Finally combine the color image produced from step 1 with edge mask produced from step 4.
<p align="center">
  <img src="https://analyticsindiamag.com/wp-content/uploads/2020/08/432a6b258bfa7df163a88bed81255db6.jpg" width="350" title="hover text">
</p>
