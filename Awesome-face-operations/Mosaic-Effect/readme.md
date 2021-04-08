# [Mosaic Effect  In Python Using OpenCV](https://github.com/Vi1234sh12/Face-X/edit/master/Awesome-face-operations/Mosaic-Effect/readme.md)
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/mosaic.png" height="100%" align="right"/>

## Introduction:
A photomosaic is an image split into a grid of rectangles, with each replaced by another image that matches the target (the image you ultimately want to appear in the photomosaic). In other words, if you look at a photomosaic from a distance, you see the target image; but if you come closer, you will see that the image actually consists of many smaller images. This works because of how the human eye works.

## History of Photomosaic: 
 Registration and mosaicing of images have been in practice since long before the age of digital computers. Shortly after the photographic process was developed in 1839, the use of photographs was demonstrated on topographical mapping . Images acquired from hill-tops or balloons were manually pieced together. After the development of airplane technology 1903 aerophotography became an exciting new field. The limited flying heights of the early airplanes and the need for large photo-maps, forced imaging experts to construct mosaic images from overlapping photographs. This was initially done by manually mosaicing images which were acquired by calibrated equipment. The need for mosaicing continued to increase later in history as satellites started sending pictures back to earth. Improvements in computer technology became a natural motivation to develop computational techniques and to solve related problems.
 
## The problem of image mosaicing is a combination of three problems:
- Correcting geometric deformations using image data and/or camera models.
- Image registration using image data and/or camera models.
- Eliminating seams from image mosaics.

## Mosaic Image Generator I/O.
 a `photographic mosaic`, also known under the term Photomosaic (a `portmanteau` of photo and `mosaic`), is a picture (usually a photograph) that has been divided into `usually equal sized` tiled sections, each of which is replaced with another photograph that matches the target photo. When viewed at low magnifications, the individual pixels appear as the primary image, while close examination reveals that the image is in fact made up of many hundreds or thousands of smaller images. Most of the time they are a computer-created type of montage.
 <br></br>
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/Mosaic3.png" align="right" width="650" height="360px"/>
There are two kinds of mosaic, depending on how the matching is done. In the simpler kind, each part of the target image is averaged down to a single color. Each of the library images is also reduced to a single color. Each part of the target image is then replaced with one from the library where these colors are as similar as possible. In effect, the target image is reduced in resolution , and then each of the resulting pixels is replaced with an image whose average color matches that pixel.

## Generating the Mosaic Image
Given the average RGB dataset and the target image, the first thing we have to do is generating a list of relevant source image filenames for each of the target image’s pixels.
 We can simply measure the RMSE `Root Mean Squared Error` between the RGB vector of each target image’s pixel with the RGB vector from our database. Then, choose the one with the lowest `RMSE` value. 
     There’s also a way to optimize our method when measuring the relevancy of source images and the pixel ‘batch’. We can filter out data points in our average RGB database which has a ‘too different’ RGB value with the pixel `batch` average RGB value.
 
## Splitting the images into tiles

Now let’s look at how to calculate the coordinates for a single tile from this grid. The tile with index (i, j) has a top-left corner coordinate of (i*w, i*j) and a bottom-right corner coordinate of `((i+1)*w, (j+1)*h)`, where w and h stand for the width and height of a tile, respectively. These can be used with the PIL to crop and create a tile from this image.

### 1.Averaging Color Values

Every pixel in an image has a color that can be represented by its red, green, and blue values. In this case, you are using 8-bit images, so each of these components has an 8-bit value in the range [0, 255]. Given an image with a total of N pixels, the average RGB is calculated as follows:

`\left ( r,g,b \right )_{avg}=\left ( \frac{\left ( r_{1} + r_{2} +....+ r_{N} \right )}{N}, \frac{\left ( g_{1} + g_{2} +....+ g_{N} \right )}{N}, \frac{\left ( b_{1} + b_{2} +....+ b_{N} \right )}{N} \right )`

`D_{1, 2}=\sqrt{\left ( r_{1} - r_{2} \right )^{2} + \left ( g_{1} - g_{2} \right )^{2} + \left ( b_{1} - b_{2} \right )^{2}}`


### 2.Matching Images
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/pyramid.png" align="right"/>
For each tile in the target image, you need to find a matching image from the images in the input folder specified by the user. To determine whether two images match, use the average RGB values. The closest match is the image with the closest average RGB value.

### The process of creating a panoramic image consists of the following steps. 
 - Detect keypoints and descriptors
 -  Detect a set of matching points that is present in both images (overlapping area)
 - Apply the RANSAC method to improve the matching process detection
 - Apply perspective transformation on one image using the other image as a reference frame
 - Stitch images togethe


### Code Overview : 

```
import cv2
def do_mosaic (frame, x, y, w, h, neighbor=9):
  fh, fw=frame.shape [0], frame.shape [1]
  if (y + h>fh) or (x + w>fw):
    return
  for i in range (0, h-neighbor, neighbor):#keypoint 0 minus neightbour to prevent overflow
    for j in range (0, w-neighbor, neighbor):
      rect=[j + x, i + y, neighbor, neighbor]
      color=frame [i + y] [j + x] .tolist () #key point 1 tolist
      left_up=(rect [0], rect [1])
      right_down=(rect [0] + neighbor-1, rect [1] + neighbor-1) #keypoint 2 minus one pixel
      cv2.rectangle (frame, left_up, right_down, color, -1)
im=cv2.imread ("test.jpg", 1)
do_mosaic (im, 219, 61, 460-219, 412-61)
while 1:
  k=cv2.waitkey (10)
  if k == 27:
    break
  cv2.imshow ("mosaic", im)
  
```


## Results Obtained :
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/mark-zuckerberg.jpg" hight="350px" />

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/images1.png" height="450px" align="left"/>
<p style="clear:both;">
<h1><a name="contributing"></a><a name="community"></a> <a href="https://github.com/akshitagupta15june/Face-X">Community</a> and <a href="https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md">Contributing</a></h1>
<p>Please do! Contributions, updates, <a href="https://github.com/akshitagupta15june/Face-X/issues"></a> and <a href=" ">pull requests</a> are welcome. This project is community-built and welcomes collaboration. Contributors are expected to adhere to the <a href="https://gssoc.girlscript.tech/">GOSSC Code of Conduct</a>.
</p>
<p>
Jump into our <a href="https://discord.com/invite/Jmc97prqjb">Discord</a>! Our projects are community-built and welcome collaboration. 👍Be sure to see the <a href="https://github.com/akshitagupta15june/Face-X/blob/master/Readme.md">Face-X Community Welcome Guide</a> for a tour of resources available to you.
</p>
<p>
<i>Not sure where to start?</i> Grab an open issue with the <a href="https://github.com/akshitagupta15june/Face-X/issues">help-wanted label</a>
</p>

**Open Source First**

 best practices for managing all aspects of distributed services. Our shared commitment to the open-source spirit push the Face-X community and its projects forward.</p>
