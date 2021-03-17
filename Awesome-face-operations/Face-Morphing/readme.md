### Face Morphing
This is a tool which creates a morphing effect. It takes two facial images as input and returns morphing from the first image to the second.
### Example:
![face morph](https://github.com/sudipg4112001/Face-X/blob/master/Awesome-face-operations/Face-Morphing/Images/images.jpg)
### Steps:
- Find point-to-point correspondences between the two images.
- Find the Delaunay Triangulation for the average of these points.
- Using these corresponding triangles in both initial and final images, perform Warping and Alpha Blending and obtain intermediate images. 
