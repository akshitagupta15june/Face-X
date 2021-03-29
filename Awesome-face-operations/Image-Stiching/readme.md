
### What is image stitching?
At the beginning of the stitching process, as input, we have several images with overlapping areas. The output is a unification of these images. It is important to note that a full scene from the input image must be preserved in the process.

### Example:
![image stiching](https://github.com/sudipg4112001/Face-X/blob/master/Awesome-face-operations/Image-Stiching/Sample-img.jpg)

### Our panorama stitching algorithm consists of four steps:

- Step #1: Detect keypoints (DoG, Harris, etc.) and extract local invariant descriptors (SIFT, SURF, etc.) from the two input images.
- Step #2: Match the descriptors between the two images.
- Step #3: Use the RANSAC algorithm to estimate a homography matrix using our matched feature vectors.
- Step #4: Apply a warping transformation using the homography matrix obtained from Step #3.
