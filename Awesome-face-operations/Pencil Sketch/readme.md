Pencil Sketch using OpenCV:

Step-1:Import all the required libraries.

Step-2:Using imread() function,we will read the image that
is to be processed.

Step-3:Using cvtColor() function,we will convert the input image
into equivalent grey-scale image.

Step-4:bitwise_not() function is used to make brighter regions
lighter and lighter regions darker so that we could find edges to
create a pencil sketch.

Step-5: Smoothing of the image also called as blurring is done using
GaussianBlur() function.

Step-6:dodgeV2() function is used to divide the grey-scale value of
image by the inverse of blurred image which highlights the sharpest
edges.

Step-7: Final output is the Pencil Sketch of input image.

Here are some of the sample output images.
![IMG_20210316_203110](https://user-images.githubusercontent.com/58857630/111414428-021de800-8706-11eb-8298-54a0d9ef236b.png)

![IMG_20210316_203217](https://user-images.githubusercontent.com/58857630/111414499-1c57c600-8706-11eb-8ac1-e38e26258aea.png)

![IMG_20210316_203144](https://user-images.githubusercontent.com/58857630/111414546-2ed1ff80-8706-11eb-8653-b98db0231d5c.png)
