# Implementing FaceSwap using OpenCV
  Here we basically use two images source image and destination image.
  <p><a><img src="https://user-images.githubusercontent.com/58718316/163699615-e0ba6b5d-9302-4342-8968-531b81c38d93.jpg" height=300 width= 400></a></p>
  <a><img src="https://user-images.githubusercontent.com/58718316/163699618-6b5d2cb1-9c66-4006-b58c-64801041b21b.jpg" height=300 width= 400></a>

  
## Using FaceSwap Step by Step:

<p>1. Face Alignment : To replace one face with another, we first need place one face approximately on top of the other so that it covers the face.</p>

<p>2. Facial Landmark Detection The geometry of the two faces are very different and so we need  to warp the source face a bit so that it covers the target face, but we also want to make sure we do not warp it beyond recognition.To achieve this we first detect facial landmarks on both images using dlib. However, unlike in Face Morphing, we do not and should not use all the points for face alignment. We simply need the points on the outer boundary of the face as show in the image.</p>

<p>3.Find Convex Hull In Computer Vision and Math jargon, the boundary of a collection of points or shape is called a “hull”. A boundary that does not have any concavities is called a “Convex Hull”. </p>

<p>4.Delaunay Triangulation The next step in alignment is to do a Delaunay triangulation of the points on the convex hull.</p>

<p>5.Affine warp triangles The final steps of face alignment to to consider corresponding triangles between the source face and the target face, and affine warp the source face triangle onto the target face. Aligning of the face and slapping one face on top of the other hardly looks unnatural. The seams are visible because of lighting and skin tone differences between the two images. The next step shows how to seamlessly combine the two images.</p>

<p>6.Seamless Cloning : Good technical ideas are like good magic. Good magicians use a combination of physics, psychology and good old sleight of hand to achieve the incredible. Image warping alone looks pretty bad. Combine it with Seamless Cloning and the results are magical</p>

 seamlessly clone parts allows of the source image onto a destination image. 
 <a href="https://learnopencv.com/face-swap-using-opencv-c-python/">Read more</a>
 
 
## Dependaries

 - Numpy

 - OpenCV 

## Output image

<a><img src="https://user-images.githubusercontent.com/58718316/163699475-ef746398-a3f4-4afb-a6b0-75453cf1de64.PNG"></a>

