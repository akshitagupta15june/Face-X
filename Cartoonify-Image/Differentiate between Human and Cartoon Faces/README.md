# Differentiate between Cartoon and Human Faces

One way to discriminate between cartoon and natural scene images is to compare a given image to its "smoothed" self. The motivation behind this is that a "smoothed" cartoon image statistically will not change much, where as a natural scene image will. In other words, take an image, cartoonify (i.e. smooth) it and subtract the result from the original.

This difference (i.e. taking its mean value) will give the level of change caused by the smoothing. The index should be high for non-smooth original (natural scene) images and low for smooth original (cartoony) images.

Smoothing/Cartoonifying is done with bilateral filtering.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/cartoon/Cartoonify-Image/Differentiate%20between%20Human%20and%20Cartoon%20Faces/Media/image.webp?raw=true" width="400" title="hover text">
  <img src="https://github.com/shireenchand/Face-X/blob/cartoon/Cartoonify-Image/Differentiate%20between%20Human%20and%20Cartoon%20Faces/Media/new.jpeg?raw=true" width="400" alt="accessibility text">
</p>


As for subtracting the cartoonyfied image from the original, it is done with the Hue channel of the HSV images. This means, first the images are converted from BGR to HSV and then subtracted.
