### METHODOLOGY:
-  Image Acquisition:
The input image to the system can be captured using a web cam or can be acquired from the hard disk. This image undergoes image enhancement, where tone mapping is applied to images with low contrast to restore the original contrast of the image.
-  Pre-processing:
 Pre-Processing plays a key role in overall process. PreProcessing stage enhances the quality of input image and locates data of interest by removing noise and smoothing the image. It removes redundancy from image without the image detail. Pre-Processing also includes filtering and normalization of image
which produces uniform size and rotated image. 
- Segmentation:
Segmentation separates image into meaningful reasons. Segmentation of an image is a method of dividing the image into homogenous, self-consistent regions corresponding to different objects in the image on the bases of texture, edge and intensity. 
-  Feature extraction:
 The facial image obtained from the face detection stage forms an input to the feature extraction stage. To obtain real time performance and to reduce time complexity, for the intent of expression recognition, only eyes and mouth are considered. The combination of two features is adequate to convey emotions accurately. 
-  Emotion classification:
 The extracted feature points are processed to obtain the inputs for support vector machine for efficient training. 
 
 ### Example:
 ![Screenshot 2021-04-21 195649](https://user-images.githubusercontent.com/60208804/115570597-dfba5400-a2db-11eb-878c-6d8793fdf44a.png)

After this the classified expression acts as an input and is used to select an appropriate playlist from the initially generated playlists and the songs from the playlists are played.
