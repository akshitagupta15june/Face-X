# Image Pyramids
  
## Introduction<br>

* An image pyramid is a collection of images - all arising from a single original image - that are successively downsampled until some desired stopping point is reached.<br>
#### There are two common kinds of image pyramids<br>
## Gaussian pyramid<br>
* Used to downsample images<br>
* Representing image in multiple scales
* They are called pyramids because the processed image can be represented in form of pyramids of different size(2^n+1)

<p align="center">
<img src="https://user-images.githubusercontent.com/74819092/119258909-5bfccb80-bbe9-11eb-9154-c73ca5e570f9.png" height=250>

<img src="https://user-images.githubusercontent.com/74819092/119236366-a1bb8480-bb54-11eb-9bcf-8dcdec71b6e3.png" height=250>
<img src="https://user-images.githubusercontent.com/74819092/119236105-3ae99b80-bb53-11eb-8ee6-ba80eac9fca6.png" width=350>
</p>


### How Algorithms Works? <br>
It works on Aliasing "Aliasing is an effect that causes different signals to become indistinguishable ".<br>
Image I of size (2N+1)x(2N+1) Output: Images g <br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74819092/119259639-c82cfe80-bbec-11eb-868b-07c685211477.png" height=400, width=500>
  </p>
<br>
The whole pyramid is only 4/3 the size of the original image.<br>
#### To	generate a Gaussian	pyramid, iterate	between	these	two	steps:	
* Smoothing: Remove high-frequency components	that could	cause	aliasing.	<br>
Smoothing	can	be achieved	by averaging	neighboring	pixels.	
The	strength of a	smoothing	operator is proportional to the	number of	pixels it averages.	
Averaging	can	be computed	as the Cross-Correlation of the	image	with a constant kernel,
Cross correlation in 1D can be computed using Matrix multiplication

* Down-sampling: Reduce	the	image	size by	½	at	each	level.



### Advantages<br>
* To make smoothed copies of images at different scales.
* highly redundant, coarse scales provide much of the information
in the finer scales

### Disadvantages<br>


### Applications <br>
* Scale	invariant	template	matching	(like	faces)	
* Progressive image	transmission	
* Image	blending	
* Used for multi-scale edge estimation
* Efficient	feature	search<br>
Some Good example of image blending <br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74819092/119258650-3c18d800-bbe8-11eb-80ad-ae383fe0fa21.png"  height=400>
  </p>



### References <br>
https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html<br>
http://www.cs.toronto.edu/~jepson/csc320/notes/pyramids.pdf<br>
https://inst.eecs.berkeley.edu/~cs194-26/fa17/Lectures/FilteringGaussianPyramids.pdf<br>
https://www.youtube.com/watch?v=1YjC9sTm0vM<br>
https://www.youtube.com/watch?v=8yvln2atFkA<br>




# Laplacian pyramid<br>
* Used to reconstruct an upsampled image from an image lower in the pyramid (with less resolution)<br>
* The Laplacian pyramid provides a coarse representation of the image as well as a set of detail images (bandpass components) at different scales.
<p align="center">
<img src="https://user-images.githubusercontent.com/74819092/119262644-8c4c6600-bbf9-11eb-887b-13108930bc43.png">

</p>

### How Algorithms Works? <br>
Over-complete decomposition based on difference-of-lowpass filters;
the image is recursively decomposed into low-pass and highpass bands.
* Each band of the Laplacian pyramid is the difference between two
adjacent low-pass images of the Gaussian pyramid,<br>
![image](https://user-images.githubusercontent.com/74819092/119261738-5659b280-bbf6-11eb-852a-10d93db93968.png)
<br>
![image](https://user-images.githubusercontent.com/74819092/119263018-00d3d480-bbfb-11eb-9762-37ddec75357c.png)



### Advantages<br>
* Eliminates blocking artifacts of JPEG at low frequencies because
of the overlapping basis functions.
* approach also allows for progressive transmission, since low-pass
representations are reasonable approximations to the image.
* coding and image reconstruction are simple

### Disadvantages<br>


### Applications <br>
* A Laplacian filter is an edge detector used to compute the second derivatives of an image, measuring the rate at which the first derivatives change.<br>
* Laplacian operator is to restore fine detail to an image which has been smoothed to remove noise. <br>
* Applies a Laplacian operator to the grayscale image and stores the output image.<br>

### References <br>
https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html<br>
http://www.cs.toronto.edu/~jepson/csc320/notes/pyramids.pdf<br>
https://inst.eecs.berkeley.edu/~cs194-26/fa17/Lectures/FilteringGaussianPyramids.pdf<br>
https://www.youtube.com/watch?v=aDY4aBLFOIg<br>
https://www.youtube.com/watch?v=_wZeX_35Iew<br>
