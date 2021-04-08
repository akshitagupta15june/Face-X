# [Mosaic Effect  In Python Using OpenCV](https://github.com/Vi1234sh12/Face-X/edit/master/Awesome-face-operations/Mosaic-Effect/readme.md)
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/mosaic.png" height="100%" align="right"/>

### Introduction:
A photomosaic is an image split into a grid of rectangles, with each replaced by another image that matches the target (the image you ultimately want to appear in the photomosaic). In other words, if you look at a photomosaic from a distance, you see the target image; but if you come closer, you will see that the image actually consists of many smaller images. This works because of how the human eye works.

### History of Photomosaic: 
 Registration and mosaicing of images have been in practice since long before the age of digital computers. Shortly after the photographic process was developed in 1839, the use of photographs was demonstrated on topographical mapping . Images acquired from hill-tops or balloons were manually pieced together. After the development of airplane technology 1903 aerophotography became an exciting new field. The limited flying heights of the early airplanes and the need for large photo-maps, forced imaging experts to construct mosaic images from overlapping photographs. This was initially done by manually mosaicing images which were acquired by calibrated equipment. The need for mosaicing continued to increase later in history as satellites started sending pictures back to earth. Improvements in computer technology became a natural motivation to develop computational techniques and to solve related problems.
 
The problem of image mosaicing is a combination of three problems:
- Correcting geometric deformations using image data and/or camera models.
- Image registration using image data and/or camera models.
- Eliminating seams from image mosaics.

### Steps:
- Read the tile images, which will replace the tiles in the original image.
- Read the target image and split it into an M×N grid of tiles.
- For each tile, fnd the best match from the input images.
- Create the fnal mosaic by arranging the selected input images in an M×N grid.

### Mosaic Image Generator I/O.
 a `photographic mosaic`, also known under the term Photomosaic (a `portmanteau` of photo and `mosaic`), is a picture (usually a photograph) that has been divided into `usually equal sized` tiled sections, each of which is replaced with another photograph that matches the target photo. When viewed at low magnifications, the individual pixels appear as the primary image, while close examination reveals that the image is in fact made up of many hundreds or thousands of smaller images. Most of the time they are a computer-created type of montage.
 <br></br>
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/Mosaic3.png" align="left" width="650" height="390px"/>
There are two kinds of mosaic, depending on how the matching is done. In the simpler kind, each part of the target image is averaged down to a single color. Each of the library images is also reduced to a single color. Each part of the target image is then replaced with one from the library where these colors are as similar as possible. In effect, the target image is reduced in resolution , and then each of the resulting pixels is replaced with an image whose average color matches that pixel.

<br></br>
### Generating the Mosaic Image
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/Pixel_Batching.png" height="300" width="650" align="right" />
Given the average RGB dataset and the target image, the first thing we have to do is generating a list of relevant source image filenames for each of the target image’s pixels.
 We can simply measure the RMSE `Root Mean Squared Error` between the RGB vector of each target image’s pixel with the RGB vector from our database. Then, choose the one with the lowest `RMSE` value. 
     There’s also a way to optimize our method when measuring the relevancy of source images and the pixel ‘batch’. We can filter out data points in our average RGB database which has a ‘too different’ RGB value with the pixel `batch` average RGB value.


<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/images1.png" align="left" height="450px" />


