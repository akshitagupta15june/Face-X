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
