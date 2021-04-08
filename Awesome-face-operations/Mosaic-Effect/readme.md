# [Mosaic Effect  In Python Using OpenCV](https://github.com/Vi1234sh12/Face-X/edit/master/Awesome-face-operations/Mosaic-Effect/readme.md)
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/mosaic.png" height="100%" align="right"/>

### Introduction:
A photomosaic is an image split into a grid of rectangles, with each replaced by another image that matches the target (the image you ultimately want to appear in the photomosaic). In other words, if you look at a photomosaic from a distance, you see the target image; but if you come closer, you will see that the image actually consists of many smaller images. This works because of how the human eye works.

### Steps:
- Read the tile images, which will replace the tiles in the original image.
- Read the target image and split it into an M×N grid of tiles.
- For each tile, fnd the best match from the input images.
- Create the fnal mosaic by arranging the selected input images in an M×N grid.

### Mosaic Image Generator I/O.
 a `photographic mosaic`, also known under the term Photomosaic (a `portmanteau` of photo and `mosaic`), is a picture (usually a photograph) that has been divided into `usually equal sized` tiled sections, each of which is replaced with another photograph that matches the target photo. When viewed at low magnifications, the individual pixels appear as the primary image, while close examination reveals that the image is in fact made up of many hundreds or thousands of smaller images. Most of the time they are a computer-created type of montage.
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/Mosaic3.png" align="center" height=""/>
There are two kinds of mosaic, depending on how the matching is done. In the simpler kind, each part of the target image is averaged down to a single color. Each of the library images is also reduced to a single color. Each part of the target image is then replaced with one from the library where these colors are as similar as possible. In effect, the target image is reduced in resolution , and then each of the resulting pixels is replaced with an image whose average color matches that pixel.

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic-Images/images1.png" align="left" height="450px" />


