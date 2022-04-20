# Implementing FaceSwap using OpenCV
<img src="https://user-images.githubusercontent.com/58718316/163847635-f5f2ac78-ed57-4361-ab08-39367d2dddf6.png" height=300 width=400 align=right>

Here we basically use two images source image and destination image. The “source image” is the one we take the face from and “destination image” is where we put the face extracted from the source image.<a href="https://learnopencv.com/face-swap-using-opencv-c-python/">Read more</a>
  
# Steps used for this project:
  #### 1 Taking two images – one as the source and another as a destination.
  #### 2 Using the dlib landmark detector on both these images. 
  #### 3 Joining the dots in the landmark detector to form triangles. 
  #### 4 Extracting these triangles
  #### 5 Placing the source image on the destination
  #### 6 Smoothening the face
  
<p><a href="https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/">Read more</a></p>
  
## Using FaceSwap Step by Step:
  

<p> 1. Face Alignment : To replace one face with another, we first need place one face approximately on top of the other so that it covers the face. The “source image” is the one we take the face from and “destination image” is where we put the face extracted from the source image.</p>

   Source image         | Destination image
 :--------------------: |:--------------------:
 **<img src="https://user-images.githubusercontent.com/58718316/163699615-e0ba6b5d-9302-4342-8968-531b81c38d93.jpg" height=300 width= 400>** | **<img src="https://user-images.githubusercontent.com/58718316/163699618-6b5d2cb1-9c66-4006-b58c-64801041b21b.jpg" height=300 width= 400>**|


<p>2. Facial Landmark Detection The geometry of the two faces are very different and so we need  to warp the source face a bit so that it covers the target face, but we also want to make sure we do not warp it beyond recognition.To achieve this we first detect facial landmarks on both images using dlib. However, unlike in Face Morphing, we do not and should not use all the points for face alignment. We simply need the points on the outer boundary of the face as show in the image.
Landmark points (dlib library)
We use the dlib library to detect the facial landmark points.
In the code below I show how to find the landmark points.<a href="https://learnopencv.com/facial-landmark-detection/">Read more</a></p>

<p>In this specific code I’m showing as example I’m detecting the landmarks of the source image, you need to apply that also to the destination image.</p>
<img src="https://user-images.githubusercontent.com/58718316/163850891-1f19439b-9efb-4664-be05-e1e771be029a.png">
<img src="https://user-images.githubusercontent.com/58718316/163850908-bcd77b1c-0a59-46a9-9421-6b1d56961722.png">



<p>3.Find Convex Hull In Computer Vision and Math jargon, the boundary of a collection of points or shape is called a “hull”. A boundary that does not have any concavities is called a “Convex Hull”.<a href="https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/">Read more</a></p>

<p>4.Delaunay Triangulation The next step in alignment is to do a Delaunay triangulation of the points on the convex hull.Delaunay Triangulation provides triangles that can be made on the image which can be used for making a polygon. In our case, the polygon is a face, and the regions on the face are divided using triangles.Since the location of the points on the faces of the two images is the same, we can extract a triangle from the first image and re-modify it according to the triangle in the second face, and apply the same technique on all the triangles to make a face that is similar to the face that is being swapped.For this process, we create a for loop that can loop over all the triangle index pairs, and then using the same index pair at an instant we draw a bounding box around the triangle. This triangle consists of the pixels of the image that lie under the triangle area.Once the bounding boxes are drawn on the triangles of both faces, we need to crop the triangle. Since the triangle does not fit exactly into the bounding box rectangle, the extra area that lies in the non-intersecting region of the bounding box and triangle needs to be removed. Thus we perform cropping the triangle.We can crop a triangle by performing a bitwise_and operation between the image and its mask.Once the mask of the same dimensions is created, a triangle that has the same shape and coordinates has to be drawn in the mask. To draw it we can use the convex hull method and fill the entire interior region of the polygon with white color.
  
<img src="https://user-images.githubusercontent.com/58718316/163852005-cdd550e9-9508-4f86-b5d3-2d8dc9fef22f.png" height=400 width=500 align=right>

Now the mask is ready and it can be used to crop the triangle. The same process goes for the triangle in the second image too. Such that at the end of the loop we can apply affine transformations on the first triangle to make the dimensions of the first triangle to be equal to the second triangle.

In the same loop, we create a replica of the second image with only zero as pixel intensities but with the same dimensions. The triangle after going through affine transformations is known as a warped triangle and has to be added to the new blank image that was created.

We need to remember a point that the warped triangle has to be pasted at the exact coordinates of the similar triangle of the second image. This can be done using the coordinates of the rectangle we are handling in the second image at that particular loop.

As we keep on looping over all triangle pairs, we can construct the face that has the properties of the second image, such as same cheek size, same nose length, etc.,<a href="https://pythonwife.com/swap-faces-using-opencv-dlib/">Read more</a></p>




<p>5.Affine warp triangles The final steps of face alignment to to consider corresponding triangles between the source face and the target face, and affine warp the source face triangle onto the target face. Aligning of the face and slapping one face on top of the other hardly looks unnatural. The seams are visible because of lighting and skin tone differences between the two images. The next step shows how to seamlessly combine the two images.<a href="https://www.analyticsvidhya.com/blog/2021/10/face-mesh-application-using-opencv-and-dlib/">Read more</a></p>

<img src="https://user-images.githubusercontent.com/58718316/163852769-671d112a-c444-4998-9642-cbcbfc45a1c5.png">


<p>6.Seamless Cloning : Good technical ideas are like good magic. Good magicians use a combination of physics, psychology and good old sleight of hand to achieve the incredible. Image warping alone looks pretty bad. Combine it with Seamless Cloning and the results are magical</p>

 seamlessly clone parts allows of the source image onto a destination image. 
 <a href="https://learnopencv.com/face-swap-using-opencv-c-python/">Read more</a>
 
 
## Dependaries

 - Numpy  <a href="https://www.w3schools.com/python/numpy/numpy_intro.asp#:~:text=NumPy%20is%20a%20Python%20library,you%20can%20use%20it%20freely.">Read more</a>

 - OpenCV <a href="https://docs.opencv.org/4.x/d9/df8/tutorial_root.html">Read more</a>

 - Dlib 68 <a href="https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/">Read more</a>

## Output image

<a><img src="https://user-images.githubusercontent.com/58718316/163699475-ef746398-a3f4-4afb-a6b0-75453cf1de64.PNG"></a>
